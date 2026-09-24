"""Fork: the dashboard, its cookies and the OIDC sign-in on several public origins.

``dashboard.public_url`` names the primary origin and ``dashboard.public_urls`` the others
(Hermie Web on its own domain). Contracts pinned here:

* the Host guard accepts every listed hostname on any port, as upstream does for its one URL;
* the WebSocket Origin check accepts a listed origin exactly (scheme + host + port);
* the write-request Origin check runs by default only with two or more origins;
* an OIDC sign-in started on an origin gets its callback on that origin, so the PKCE cookie and
  the session cookie never leave it -- and an unlisted Host is never reflected anywhere;
* ``X-Forwarded-Host`` counts only from a trusted proxy, and only when it names a listed origin;
* a single ``public_url``, and today's Hermie Web (Host/Origin rewritten to it), work as before,
  including the three proxy shapes a review reproduced against the first version.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.base import InvalidCodeError, InvalidCredentialsError, Session
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider, _sign

A = "https://hermes.example.test"
B = "https://app.example.test"


class RecordingIdP(StubAuthProvider):
    """Stub IdP that, like a real one, redeems a code only for the redirect_uri it authorized."""

    name = "rec"

    def __init__(self):
        super().__init__()
        self.authorized: list[str] = []
        self.redeemed: list[str] = []

    def start_login(self, *, redirect_uri):
        self.authorized.append(redirect_uri)
        return super().start_login(redirect_uri=redirect_uri)

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        self.redeemed.append(redirect_uri)
        if redirect_uri not in self.authorized:
            raise InvalidCodeError("redirect_uri mismatch")
        return super().complete_login(
            code=code, state=state, code_verifier=code_verifier, redirect_uri=redirect_uri)


class PasswordIdP(StubAuthProvider):
    """A password provider (the shape of the bundled basic-auth one)."""

    name = "pw"
    supports_password = True

    def complete_password_login(self, *, username, password):
        if (username, password) != ("robin", "pw"):
            raise InvalidCredentialsError("bad credentials")
        exp = int(time.time()) + 3600
        claims = {"sub": "robin", "email": "", "name": "Robin", "org_id": "", "exp": exp}
        return Session(user_id="robin", email="", display_name="Robin", org_id="", provider=self.name,
                       expires_at=exp, access_token=_sign(claims), refresh_token="")


@pytest.fixture
def idp():
    clear_providers()
    provider = RecordingIdP()
    register_provider(provider)
    register_provider(PasswordIdP())
    yield provider
    clear_providers()


@pytest.fixture
def deploy(monkeypatch, idp):
    """Configure the dashboard the way ``start_server`` does, for a given ``dashboard`` block."""

    def _deploy(dashboard: dict, *, bound: str = "127.0.0.1") -> None:
        from hermes_cli.dashboard_auth import origins

        origins._warned_refused.clear()  # each test sees its own once-per-Origin warnings
        monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"dashboard": dashboard})
        for name in ("auth_required", "bound_host", "trusted_public_hosts", "public_origins",
                     "write_origin_check"):
            monkeypatch.setattr(
                web_server.app.state, name, getattr(web_server.app.state, name, None), raising=False)
        web_server.app.state.bound_host = bound
        web_server._configure_auth_gate(bound, False, None, None)

    return _deploy


def _jar(client) -> dict[str, set[str]]:
    """Cookie names per host the browser (the test client's jar) holds them for."""
    out: dict[str, set[str]] = {}
    for cookie in client.cookies.jar:
        out.setdefault(cookie.domain, set()).add(cookie.name)
    return out


def _sign_in(client, idp) -> tuple[str, "object"]:
    """Walk /auth/login -> IdP -> /auth/callback like a browser; return (redirect_uri, callback)."""
    start = client.get("/auth/login?provider=rec", follow_redirects=False)
    assert start.status_code == 302, start.text
    redirect_uri = start.headers["location"].split("?", 1)[0]
    callback = client.get(start.headers["location"], follow_redirects=False)
    return redirect_uri, callback


# --- the Host / Origin guard ----------------------------------------------------------------


def _ws_reason(host: str, origin: str):
    from hermes_cli.web_server_chat import _ws_host_origin_reason

    return _ws_host_origin_reason(SimpleNamespace(headers={"host": host, "origin": origin}))


def test_host_guard_takes_every_listed_hostname_and_websocket_origin_is_exact(deploy, caplog):
    deploy({"public_url": A, "public_urls": [B]})
    client = TestClient(web_server.app)

    def host_status(host: str) -> int:
        return client.get("/api/status", headers={"Host": host}).status_code

    # Any port: proxies rewrite it ($host, $host:$server_port behind TLS offload).
    for listed in ("hermes.example.test", "app.example.test", "app.example.test:443",
                   "app.example.test:8443", "hermes.example.test:80"):
        assert host_status(listed) != 400, listed
    for refused in ("evil.example.test", "app.example.test.evil.test"):
        assert host_status(refused) == 400, refused

    assert _ws_reason("app.example.test", B) is None
    assert _ws_reason("hermes.example.test", A) is None
    assert _ws_reason("app.example.test:8443", f"{B}:8443") is None  # the page's own host:port
    with caplog.at_level(logging.WARNING, logger="hermes_cli.dashboard_auth.origins"):
        for _ in range(2):
            scheme = _ws_reason("app.example.test", "http://app.example.test")
        port = _ws_reason("app.example.test", f"{B}:8443")
        evil = _ws_reason("app.example.test", "https://evil.example.test")
    for reason, origin in ((scheme, "http://app.example.test"), (port, f"{B}:8443"),
                           (evil, "https://evil.example.test")):
        assert reason == f"origin_mismatch origin={origin} expected={A}, {B}"
    warned = [r.getMessage() for r in caplog.records if "WebSocket upgrade refused" in r.getMessage()]
    assert len(warned) == 3  # once per refused Origin, naming it and the listed ones
    assert "'http://app.example.test'" in warned[0] and f"{A}, {B}" in warned[0]


def test_state_changing_request_needs_a_listed_or_own_origin(deploy, caplog):
    """SameSite=Lax lets a sibling subdomain's POST carry the cookie; the Origin check does not."""
    deploy({"public_url": A, "public_urls": [B]}, bound="0.0.0.0")
    client = TestClient(web_server.app, base_url=A)

    def logout(origin: str, **headers):
        return client.post("/auth/logout", headers={"Origin": origin, **headers}, follow_redirects=False)

    assert logout(A).status_code == 302
    assert logout(B).status_code == 302  # e.g. today's Hermie Web on B relaying to the gateway's host
    assert logout("http://hermes.example.test").status_code == 403  # scheme mismatch
    with caplog.at_level(logging.WARNING, logger="hermes_cli.dashboard_auth.origins"):
        refused = logout("https://evil.example.test")
        logout("https://evil.example.test")
    assert refused.status_code == 403
    assert "'https://evil.example.test'" in refused.json()["detail"]
    assert f"({A}, {B})" in refused.json()["detail"]
    assert "dashboard.public_urls" in refused.json()["detail"]
    assert "dashboard.write_origin_check: off" in refused.json()["detail"]
    assert len([r for r in caplog.records if "write request refused" in r.getMessage()
                and "evil.example.test" in r.getMessage()]) == 1  # once per refused Origin
    assert logout("https://evil.example.test", Authorization="Bearer x").status_code == 302  # not CSRF-able
    assert logout("https://evil.example.test", Authorization="Bearer ").status_code == 403  # cookie path


@pytest.mark.parametrize("dashboard,refused", [
    ({"public_url": A}, False),                                       # auto: one origin -> off
    ({"public_url": A, "write_origin_check": "on"}, True),
    ({"public_url": A, "public_urls": [B]}, True),                    # auto: two origins -> on
    ({"public_url": A, "public_urls": [B], "write_origin_check": "off"}, False),
])
def test_write_origin_check_is_on_by_default_only_with_several_origins(deploy, dashboard, refused):
    deploy(dashboard, bound="0.0.0.0")
    client = TestClient(web_server.app, base_url=A)

    status = client.post("/auth/logout", headers={"Origin": "https://evil.example.test"},
                         follow_redirects=False).status_code

    assert status == (403 if refused else 302)


# --- OIDC round trip per origin -------------------------------------------------------------


@pytest.mark.parametrize("origin,host", [(B, "app.example.test"), (A, "hermes.example.test")])
def test_sign_in_round_trips_on_the_origin_it_started_on(deploy, idp, origin, host):
    deploy({"public_url": A, "public_urls": [B]})
    client = TestClient(web_server.app, base_url=origin)

    redirect_uri, callback = _sign_in(client, idp)

    assert redirect_uri == f"{origin}/auth/callback"
    assert idp.redeemed == [redirect_uri]  # the token exchange names the same callback
    assert callback.status_code == 302, callback.text
    assert callback.headers["location"] == "/"  # same-origin landing, no loginReturn needed
    jar = _jar(client)
    assert set(jar) == {host}, jar  # PKCE and session cookies never left this host
    assert any(name.endswith("hermes_session_at") for name in jar[host])
    assert all("domain=" not in c.lower() for c in callback.headers.get_list("set-cookie"))


def test_unlisted_host_gets_the_primary_callback_and_is_never_reflected(deploy, idp):
    deploy({"public_url": A, "public_urls": [B]}, bound="0.0.0.0")  # Host guard wide open
    client = TestClient(web_server.app, base_url="https://evil.example.test")

    start = client.get("/auth/login?provider=rec", follow_redirects=False)

    assert start.headers["location"].startswith(f"{A}/auth/callback?")
    assert "evil" not in start.headers["location"]
    assert all("evil" not in c and "domain=" not in c.lower()
               for c in start.headers.get_list("set-cookie"))


# --- X-Forwarded-Host through the real uvicorn proxy chain -----------------------------------


def _behind_uvicorn(peer: str, base_url: str = A) -> TestClient:
    """The app as uvicorn serves it in gated mode, with the socket peer ``peer``."""
    import uvicorn

    from hermes_cli.dashboard_auth.origins import install_forwarded_peer_marker

    config = uvicorn.Config(web_server.app, proxy_headers=True, log_config=None,
                            forwarded_allow_ips=["127.0.0.1", "::1"])
    install_forwarded_peer_marker(config)
    return TestClient(config.loaded_app, base_url=base_url, client=(peer, 50000))


@pytest.mark.parametrize("peer,forwarded,expected", [
    ("203.0.113.9", "app.example.test", A),   # untrusted peer: header ignored
    ("127.0.0.1", "app.example.test", B),     # trusted proxy naming a listed origin
    ("127.0.0.1", "evil.example.test", A),    # trusted proxy naming an unlisted host
    ("127.0.0.1", "app.example.test:8443", A),  # listed host, unlisted port
])
def test_forwarded_host_counts_only_from_a_trusted_proxy_and_only_when_listed(
        deploy, idp, peer, forwarded, expected):
    deploy({"public_url": A, "public_urls": [B]})
    client = _behind_uvicorn(peer)

    start = client.get("/auth/login?provider=rec", follow_redirects=False,
                       headers={"X-Forwarded-Host": forwarded, "X-Forwarded-Proto": "https"})

    assert start.headers["location"].startswith(f"{expected}/auth/callback?")


def test_portless_host_picks_the_only_listed_port_for_that_host(deploy, idp):
    """nginx ``$host`` drops the port; with one listed origin for that scheme + host it is that."""
    deploy({"public_url": A, "public_urls": ["https://app.example.test:9443"]})
    client = _behind_uvicorn("127.0.0.1")

    start = client.get("/auth/login?provider=rec", follow_redirects=False,
                       headers={"X-Forwarded-Host": "app.example.test", "X-Forwarded-Proto": "https"})

    assert start.headers["location"].startswith("https://app.example.test:9443/auth/callback?")


def test_redirect_uri_uses_the_origins_the_guards_were_started_with(deploy, idp, monkeypatch):
    deploy({"public_url": A, "public_urls": [B]})
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"dashboard": {"public_url": "https://moved.example.test"}})
    client = TestClient(web_server.app, base_url=B)

    start = client.get("/auth/login?provider=rec", follow_redirects=False)

    assert start.headers["location"].startswith(f"{B}/auth/callback?")


@pytest.mark.parametrize("peer,allow,trusted", [
    ("1.2.3.4", ["127.0.0.1", "::1"], False),
    ("127.0.0.1", ["127.0.0.1", "::1"], True),
    ("10.0.0.5", ["127.0.0.1", "10.0.0.0/8"], True),
    ("1.2.3.4", "*", True),                       # uvicorn's own "*" means every peer
    ("1.2.3.4", ["127.0.0.1", "*"], False),       # ...but not inside a list, in uvicorn either
    (None, ["127.0.0.1"], False),
])
def test_forwarded_peer_marker_trusts_exactly_the_peers_uvicorn_trusts(peer, allow, trusted):
    import asyncio

    import uvicorn

    from hermes_cli.dashboard_auth.origins import TRUSTED_PEER_SCOPE_KEY, install_forwarded_peer_marker

    seen: dict = {}

    async def app(scope, receive, send):
        seen.update(scope)

    config = uvicorn.Config(app, proxy_headers=True, forwarded_allow_ips=allow, log_config=None)
    install_forwarded_peer_marker(config)
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": "/", "raw_path": b"/", "query_string": b"", "root_path": "",
             "headers": [(b"host", b"x"), (b"x-forwarded-for", b"127.0.0.1"),
                         (b"x-forwarded-proto", b"https")],
             "client": (peer, 1) if peer else None, "server": ("127.0.0.1", 9119)}
    asyncio.run(config.loaded_app(scope, None, None))

    assert seen[TRUSTED_PEER_SCOPE_KEY] is trusted
    assert (seen["scheme"] == "https") is trusted  # the marker and uvicorn agree


def test_marker_that_cannot_be_installed_is_reported(caplog):
    from hermes_cli.dashboard_auth.origins import install_forwarded_peer_marker

    changed_uvicorn = SimpleNamespace(proxy_headers=True, loaded=True, forwarded_allow_ips=["127.0.0.1"])
    with caplog.at_level(logging.WARNING, logger="hermes_cli.dashboard_auth.origins"):
        install_forwarded_peer_marker(changed_uvicorn)

    assert any("X-Forwarded-Host will be ignored" in r.getMessage() for r in caplog.records)


# --- backward compatibility -----------------------------------------------------------------


def test_single_public_url_behaves_as_before(deploy, idp):
    deploy({"public_url": "https://example.test/hermes"})
    client = TestClient(web_server.app, base_url="https://example.test")

    assert web_server.app.state.trusted_public_hosts == frozenset({"example.test"})
    for host in ("example.test", "example.test:443"):
        assert client.get("/api/status", headers={"Host": host}).status_code != 400
    start = client.get("/auth/login?provider=rec", follow_redirects=False)
    assert start.headers["location"].startswith("https://example.test/hermes/auth/callback?")


class TestSingleUrlProxyShapes:
    """Three single-``public_url`` deployments the first version of this change broke. Each
    reaches a loopback-bound gateway through a TLS proxy on the same machine."""

    OFFLOAD = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.7"}

    def test_host_with_server_port_behind_tls_offload(self, deploy, idp):
        """nginx ``proxy_set_header Host $host:$server_port`` behind TLS offload: Host ``:80``."""
        deploy({"public_url": A})
        client = _behind_uvicorn("127.0.0.1", base_url="http://127.0.0.1:9119")
        relayed = {**self.OFFLOAD, "Host": "hermes.example.test:80"}

        assert client.get("/api/status", headers=relayed).status_code == 200
        start = client.get("/auth/login?provider=rec", headers=relayed, follow_redirects=False)
        assert start.headers["location"].startswith(f"{A}/auth/callback?")
        logout = client.post("/auth/logout", headers={**relayed, "Origin": A}, follow_redirects=False)
        assert logout.status_code == 302
        assert _ws_reason("hermes.example.test:80", A) is None

    def test_public_url_written_as_http_while_the_browser_is_on_https(self, deploy, idp, caplog):
        """HTTP behaves as before; the WebSocket is refused -- the fix is to write ``https://``
        -- and the refusal is now logged, naming the Origin and what is listed."""
        deploy({"public_url": "http://hermes.example.test"})
        client = _behind_uvicorn("127.0.0.1", base_url="http://127.0.0.1:9119")
        relayed = {**self.OFFLOAD, "Host": "hermes.example.test"}

        assert client.get("/api/status", headers=relayed).status_code == 200
        start = client.get("/auth/login?provider=rec", headers=relayed, follow_redirects=False)
        assert start.headers["location"].startswith("http://hermes.example.test/auth/callback?")
        logout = client.post("/auth/logout", headers={**relayed, "Origin": A}, follow_redirects=False)
        assert logout.status_code == 302
        with caplog.at_level(logging.WARNING, logger="hermes_cli.dashboard_auth.origins"):
            reason = _ws_reason("hermes.example.test", A)
        assert reason == f"origin_mismatch origin={A} expected=http://hermes.example.test"
        assert any(A in r.getMessage() and "http://hermes.example.test" in r.getMessage()
                   for r in caplog.records)

    def test_browser_on_another_port_with_password_auth(self, deploy, idp):
        """The dashboard opened on ``:9443`` while ``public_url`` names no port."""
        deploy({"public_url": A})
        client = _behind_uvicorn("127.0.0.1", base_url="http://127.0.0.1:9119")
        relayed = {**self.OFFLOAD, "Host": "hermes.example.test:9443",
                   "Origin": "https://hermes.example.test:9443"}

        assert client.get("/api/status", headers=relayed).status_code == 200
        login = client.post("/auth/password-login", headers=relayed,
                            json={"provider": "pw", "username": "robin", "password": "pw"})
        assert login.status_code == 200, login.text
        assert any("hermes_session_at" in c for c in login.headers.get_list("set-cookie"))
        assert _ws_reason("hermes.example.test:9443", "https://hermes.example.test:9443") is None


@pytest.mark.parametrize("listed,expected", [((), A), ((B,), B)])
def test_todays_hermie_web_rewriting_host_and_origin_still_works(deploy, idp, listed, expected):
    """Hermie Web (loopback peer) rewrites Host/Origin to the primary and forwards its own host.
    Unlisted, that changes nothing; listed, the callback moves to Hermie Web's origin."""
    deploy({"public_url": A, "public_urls": list(listed)})
    client = _behind_uvicorn("127.0.0.1")
    relayed = {"Host": "hermes.example.test", "X-Forwarded-Host": "app.example.test",
               "X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.7"}

    start = client.get("/auth/login?provider=rec&next=/hermie", follow_redirects=False, headers=relayed)
    logout = client.post("/auth/logout", follow_redirects=False, headers={**relayed, "Origin": A})

    assert start.headers["location"].startswith(f"{expected}/auth/callback?")
    assert logout.status_code == 302


def test_idp_refusing_the_callback_names_the_redirect_uri_to_register(deploy, idp):
    deploy({"public_url": A, "public_urls": [B]})
    client = TestClient(web_server.app, base_url=B)
    client.get("/auth/login?provider=rec", follow_redirects=False)

    refused = client.get("/auth/callback?error=invalid_request"
                         "&error_description=redirect_uri+is+not+registered", follow_redirects=False)

    assert refused.status_code == 400
    assert f"{B}/auth/callback" in refused.json()["detail"]
