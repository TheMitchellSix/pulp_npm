# coding=utf-8
"""Tests for native npm/yarn publish (PUT /<package>) against pulp_npm registry endpoints."""

import asyncio
import base64
import io
import json
import os
import tarfile
import uuid

import aiohttp
import pytest


def _pulp_base_url():
    protocol = os.environ.get("API_PROTOCOL", "https")
    host = os.environ.get("API_HOST", "pulp")
    port = os.environ.get("API_PORT", "443")
    return f"{protocol}://{host}:{port}"


def _pulp_auth():
    return aiohttp.BasicAuth(
        os.environ.get("ADMIN_USERNAME", "admin"),
        os.environ.get("ADMIN_PASSWORD", "password"),
    )


def _build_npm_tgz(name="test-pkg", version="1.0.0", extra=None):
    """Build a minimal npm .tgz tarball in-memory and return bytes.

    ``extra`` is merged into the tarball's ``package.json``, so tests can assert that
    fields beyond ``name``/``version`` (e.g. ``bin``, ``dependencies``) survive being
    served back out through the registry's metadata endpoint.
    """
    manifest = {"name": name, "version": version, **(extra or {})}
    package_json = json.dumps(manifest).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="package/package.json")
        info.size = len(package_json)
        tar.addfile(info, io.BytesIO(package_json))
    buf.seek(0)
    return buf.read()


def _build_publish_body(name, version, tgz_bytes, extra=None):
    """Build the JSON body that npm/yarn sends on ``PUT /<package>``."""
    base_name = name.split("/")[-1] if "/" in name else name
    tarball_filename = f"{base_name}-{version}.tgz"
    return {
        "_id": name,
        "name": name,
        "dist-tags": {"latest": version},
        "versions": {
            version: {
                "name": name,
                "version": version,
                "dist": {"tarball": f"{name}/-/{tarball_filename}"},
                **(extra or {}),
            }
        },
        "_attachments": {
            tarball_filename: {
                "content_type": "application/octet-stream",
                "data": base64.b64encode(tgz_bytes).decode(),
                "length": len(tgz_bytes),
            }
        },
    }


def _npm_publish_url(base_path, package_name, domain=None):
    """Build the URL for a PUT publish request."""
    escaped = package_name.replace("/", "%2F")
    if domain:
        return f"{_pulp_base_url()}/npm/{domain}/{base_path}/{escaped}"
    return f"{_pulp_base_url()}/npm/{base_path}/{escaped}"


def _npm_meta_url(base_path, suffix, domain=None):
    """Build a URL for registry meta-endpoints (``-/ping``, ``-/whoami``, etc.)."""
    if domain:
        return f"{_pulp_base_url()}/npm/{domain}/{base_path}/{suffix}"
    return f"{_pulp_base_url()}/npm/{base_path}/{suffix}"


def _run(coro):
    return asyncio.run(coro)


async def _put_publish(url, body, auth=None):
    async with aiohttp.ClientSession(auth=auth or _pulp_auth()) as session:
        async with session.put(url, json=body, ssl=False) as resp:
            text = await resp.text()
            return resp.status, text


async def _get(url, auth=None):
    async with aiohttp.ClientSession(auth=auth or _pulp_auth()) as session:
        async with session.get(url, ssl=False) as resp:
            text = await resp.text()
            return resp.status, text


async def _get_no_redirect(url, auth=None):
    """GET without following redirects, returning (status, location_header, text).

    Defaults to authenticated admin creds, matching ``_get``/``_put_publish``.
    Pass ``auth=False`` explicitly to send the request with no credentials at all.
    """
    session_auth = None if auth is False else (auth or _pulp_auth())
    async with aiohttp.ClientSession(auth=session_auth) as session:
        async with session.get(url, ssl=False, allow_redirects=False) as resp:
            text = await resp.text()
            return resp.status, resp.headers.get("Location"), text


# ---------------------------------------------------------------------------
# Publish tests
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_publish_unscoped_package(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Publish an unscoped package via PUT /<package> and verify it lands in the repo."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"test-publish-{uuid.uuid4().hex[:8]}"
    version = "1.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    repo = npm_bindings.RepositoriesNpmApi.read(repo.pulp_href)
    assert repo.latest_version_href is not None
    assert repo.latest_version_href.endswith("/versions/1/")

    ver = npm_bindings.RepositoriesNpmVersionsApi.read(repo.latest_version_href)
    assert ver.content_summary.present["npm.package"]["count"] == 1


@pytest.mark.parallel
def test_publish_scoped_package(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Publish a scoped package (@scope/name) via PUT /@scope%2Fname."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    scope = uuid.uuid4().hex[:6]
    pkg_name = f"@{scope}/my-lib"
    version = "2.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    repo = npm_bindings.RepositoriesNpmApi.read(repo.pulp_href)
    ver = npm_bindings.RepositoriesNpmVersionsApi.read(repo.latest_version_href)
    assert ver.content_summary.present["npm.package"]["count"] == 1


@pytest.mark.parallel
def test_publish_duplicate_is_idempotent(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Publishing the same version twice should succeed (idempotent)."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"dup-pub-{uuid.uuid4().hex[:8]}"
    version = "1.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    s1, _ = _run(_put_publish(url, body))
    s2, _ = _run(_put_publish(url, body))
    assert s1 == 201
    assert s2 == 201


@pytest.mark.parallel
def test_publish_multiple_versions(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Publish two different versions of the same package."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"multi-ver-{uuid.uuid4().hex[:8]}"

    for ver in ("1.0.0", "2.0.0"):
        tgz = _build_npm_tgz(name=pkg_name, version=ver)
        body = _build_publish_body(pkg_name, ver, tgz)
        url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
        status, text = _run(_put_publish(url, body))
        assert status == 201, f"Expected 201 for {ver}, got {status}: {text}"

    repo = npm_bindings.RepositoriesNpmApi.read(repo.pulp_href)
    ver = npm_bindings.RepositoriesNpmVersionsApi.read(repo.latest_version_href)
    assert ver.content_summary.present["npm.package"]["count"] == 2


@pytest.mark.parallel
def test_publish_without_repository_fails(
    npm_bindings,
    npm_distribution_factory,
    pulp_settings,
):
    """Publishing to a distribution with no repository should return 400."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    distro = npm_distribution_factory()

    pkg_name = f"no-repo-{uuid.uuid4().hex[:8]}"
    tgz = _build_npm_tgz(name=pkg_name, version="1.0.0")
    body = _build_publish_body(pkg_name, "1.0.0", tgz)

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, _ = _run(_put_publish(url, body))
    assert status == 400


@pytest.mark.parallel
def test_publish_missing_attachments_fails(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    pulp_settings,
):
    """A publish body with no _attachments should return 400."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"no-attach-{uuid.uuid4().hex[:8]}"
    body = {
        "name": pkg_name,
        "versions": {"1.0.0": {"name": pkg_name, "version": "1.0.0"}},
    }

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, _ = _run(_put_publish(url, body))
    assert status == 400


@pytest.mark.parallel
def test_publish_name_mismatch_fails(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    pulp_settings,
):
    """Body name != URL package name should return 400."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"url-name-{uuid.uuid4().hex[:8]}"
    tgz = _build_npm_tgz(name="wrong-body-name", version="1.0.0")
    body = _build_publish_body("wrong-body-name", "1.0.0", tgz)

    url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, _ = _run(_put_publish(url, body))
    assert status == 400


# ---------------------------------------------------------------------------
# Metadata GET (install/resolve) -- see GH #420
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_get_unscoped_package_redirects_to_content_app(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """GET /<package> on the publish endpoint should redirect to the content app.

    npm resolves a scope to a single registry URL and uses it for both
    publish and install, so the ``npm/...`` endpoint has to answer metadata
    GETs too, not just PUT.
    """
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"test-get-{uuid.uuid4().hex[:8]}"
    version = "1.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    publish_url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(publish_url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    status, location, _ = _run(_get_no_redirect(publish_url))
    assert status in (301, 302, 303, 307, 308), f"Expected a redirect, got {status}"
    expected_location = f"{distro.base_url.rstrip('/')}/{pkg_name}"
    assert location == expected_location, f"{location!r} != {expected_location!r}"

    # Following the redirect should return real npm registry metadata, generated by
    # the content app's existing NpmDistribution.content_handler().
    status, response_text = _run(_get(location))
    assert status == 200, f"Expected 200, got {status}: {response_text}"
    data = json.loads(response_text)
    assert data["name"] == pkg_name
    assert data["dist-tags"]["latest"] == version
    assert version in data["versions"]
    assert data["versions"][version]["dist"]["tarball"]


@pytest.mark.parallel
def test_get_package_metadata_includes_full_manifest(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Served per-version metadata must include the full published package.json, not just
    ``_id``/``dist`` -- npm decides whether to create a bin shim at install time from the
    registry metadata's ``bin`` field, so a truncated response silently breaks installs of
    packages with a bin entry even though the tarball itself is unaffected (see GH #420's
    follow-on: the endpoint responds, but with incomplete per-version data).
    """
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"test-manifest-{uuid.uuid4().hex[:8]}"
    version = "1.0.0"
    manifest_extra = {
        "bin": {pkg_name: "bin/cli.js"},
        "dependencies": {"semver": "^7.0.0"},
        "main": "index.js",
    }
    tgz = _build_npm_tgz(name=pkg_name, version=version, extra=manifest_extra)
    body = _build_publish_body(pkg_name, version, tgz, extra=manifest_extra)

    publish_url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(publish_url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    status, location, _ = _run(_get_no_redirect(publish_url))
    assert status in (301, 302, 303, 307, 308), f"Expected a redirect, got {status}"

    status, response_text = _run(_get(location))
    assert status == 200, f"Expected 200, got {status}: {response_text}"
    data = json.loads(response_text)

    version_data = data["versions"][version]
    assert version_data["bin"] == manifest_extra["bin"]
    assert version_data["dependencies"] == manifest_extra["dependencies"]
    assert version_data["main"] == manifest_extra["main"]
    # The served tarball URL must still point at Pulp's own content app, not whatever
    # (possibly relative/bogus) dist.tarball the publishing client sent.
    assert version_data["dist"]["tarball"].startswith(distro.base_url)


@pytest.mark.parallel
def test_get_scoped_package_redirects_to_content_app(
    npm_bindings,
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """GET /@scope%2Fpackage on the publish endpoint should redirect correctly."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    scope = uuid.uuid4().hex[:6]
    pkg_name = f"@{scope}/my-lib"
    version = "2.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    publish_url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(publish_url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    status, location, _ = _run(_get_no_redirect(publish_url))
    assert status in (301, 302, 303, 307, 308), f"Expected a redirect, got {status}"

    status, response_text = _run(_get(location))
    assert status == 200, f"Expected 200, got {status}: {response_text}"
    data = json.loads(response_text)
    assert data["name"] == pkg_name
    assert data["dist-tags"]["latest"] == version


@pytest.mark.parallel
def test_get_package_metadata_requires_no_authentication(
    npm_repository_factory,
    npm_distribution_factory,
    monitor_task,
    pulp_settings,
):
    """Metadata reads should be open (matching pulp_python's SimpleView), unlike publish.

    The redirect target (the content app) still enforces any content guard configured
    on the distribution -- this only confirms the publish endpoint itself doesn't
    require credentials to hand out that redirect.
    """
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    repo = npm_repository_factory()
    distro = npm_distribution_factory(repository=repo.pulp_href)

    pkg_name = f"test-anon-{uuid.uuid4().hex[:8]}"
    version = "1.0.0"
    tgz = _build_npm_tgz(name=pkg_name, version=version)
    body = _build_publish_body(pkg_name, version, tgz)

    publish_url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)
    status, response_text = _run(_put_publish(publish_url, body))
    assert status == 201, f"Expected 201, got {status}: {response_text}"

    status, location, _ = _run(_get_no_redirect(publish_url, auth=False))
    assert status in (301, 302, 303, 307, 308), f"Expected a redirect with no auth, got {status}"


@pytest.mark.parallel
def test_get_nonexistent_package_not_found(npm_distribution_factory, pulp_settings):
    """GET on a package that was never published should not succeed."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    distro = npm_distribution_factory()

    pkg_name = f"never-published-{uuid.uuid4().hex[:8]}"
    publish_url = _npm_publish_url(distro.base_path, pkg_name, domain=domain)

    status, location, _ = _run(_get_no_redirect(publish_url))
    assert status in (301, 302, 303, 307, 308), f"Expected a redirect, got {status}"

    status, _ = _run(_get(location))
    assert status == 404, f"Expected 404 for an unpublished package, got {status}"


# ---------------------------------------------------------------------------
# Ping / Whoami
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_ping(npm_distribution_factory, pulp_settings):
    """GET /-/ping should return 200 with empty JSON object."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    distro = npm_distribution_factory()
    url = _npm_meta_url(distro.base_path, "-/ping", domain=domain)
    status, text = _run(_get(url))
    assert status == 200
    assert json.loads(text) == {}


@pytest.mark.parallel
def test_whoami_authenticated(npm_distribution_factory, pulp_settings):
    """GET /-/whoami with valid credentials should return the username."""
    domain = "default" if pulp_settings.DOMAIN_ENABLED else None
    distro = npm_distribution_factory()
    url = _npm_meta_url(distro.base_path, "-/whoami", domain=domain)
    status, text = _run(_get(url))
    assert status == 200
    data = json.loads(text)
    assert "username" in data
    assert data["username"] == os.environ.get("ADMIN_USERNAME", "admin")
