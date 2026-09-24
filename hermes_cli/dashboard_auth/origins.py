"""The dashboard's public origins: the allow-list every Host, Origin and redirect_uri is held to.

Fork-only (fullstackstudio-org/hermes-agent). Upstream accepts ONE public URL
(``dashboard.public_url``). This fork also reads ``dashboard.public_urls``, so the same gateway
can be reached on e.g. ``https://hermes.example.com`` AND ``https://app.example.com`` (Hermie
Web on its own domain), with a working OIDC sign-in on each.

An origin is exactly ``scheme + host + port``. The rules built on it:

* **Host guard** (``web_server._is_accepted_host``): every listed HOSTNAME is accepted, on any
  port, exactly as upstream treats its one ``public_url``. Proxies rewrite the Host port too
  freely (nginx ``$host``, ``$host:$server_port`` behind TLS offload) for it to mean anything.
* **WebSocket Origin**: a web ``Origin`` naming a listed hostname must match a listed origin
  exactly (scheme, host, port), or be the request's own explicit ``host:port``.
* **Write-request Origin** (``dashboard.write_origin_check``, on by default only with two or
  more origins): a browser's ``Origin`` must be listed or the request's own.
* **redirect_uri**: built from the origin the sign-in was started on, but only when that
  origin is listed; otherwise from the primary. The value is always a listed URL, never text
  taken from the request, so an unlisted or forged Host cannot become a redirect target. It
  uses the list read at startup, the same one the guards use, so it never names a callback
  the guards would refuse.

The request's own origin is its ``Host`` (and the scheme uvicorn settled on). A proxy's
``X-Forwarded-Host`` replaces the ``Host`` only when the socket peer is a trusted proxy
(loopback or ``dashboard.trusted_proxies``) -- the same peers uvicorn already lets set
``X-Forwarded-Proto``. uvicorn rewrites ``scope["client"]`` before the app runs, so the raw
peer's trust is recorded outside it by :class:`ForwardedPeerMarker`.
"""
from __future__ import annotations

import ipaddress
import logging
import re
from typing import Iterable, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import urlparse

from hermes_cli.dashboard_auth import prefix as _prefix

_log = logging.getLogger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}
# ASGI names the scheme of a WebSocket scope ws/wss; its browser origin is http/https.
_WS_SCHEMES = {"ws": "http", "wss": "https"}
# Set on every http/websocket scope by :class:`ForwardedPeerMarker`; absent means untrusted.
TRUSTED_PEER_SCOPE_KEY = "hermes.forwarded_peer_trusted"
_warned_unmatched: set = set()

OriginKey = Tuple[str, str, int]


class PublicOrigin(NamedTuple):
    """One listed origin. ``base_url`` is the listed URL itself (path prefix included), the
    only thing a redirect_uri is ever built from."""
    scheme: str
    host: str
    port: int
    base_url: str

    @property
    def key(self) -> OriginKey:
        return (self.scheme, self.host, self.port)

    def serialize(self) -> str:
        """The browser's ``Origin`` form: default port omitted, IPv6 bracketed."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        if _DEFAULT_PORTS[self.scheme] == self.port:
            return f"{self.scheme}://{host}"
        return f"{self.scheme}://{host}:{self.port}"


def split_authority(value: str) -> Tuple[str, Optional[int]]:
    """``(hostname, port or None)`` from a ``Host``-style authority, or ``("", None)``.

    Authorities are not URLs: URL syntax, ambiguous (unbracketed) IPv6, malformed brackets and
    non-numeric or out-of-range ports are refused, so every check built on this fails closed."""
    value = (value or "").strip()
    if not value or "://" in value or any(c in value for c in '"\'<> \n\r\t/?#@\\'):
        return "", None
    port_text = ""
    if value.startswith("["):
        close = value.find("]")
        if close == -1:
            return "", None
        hostname, suffix = value[1:close], value[close + 1:]
        # Bracket notation is reserved for IPv6 literals.
        if ":" not in hostname or (suffix and not re.fullmatch(r":\d+", suffix)):
            return "", None
        port_text = suffix[1:]
    elif value.count(":") > 1:
        # Unbracketed IPv6 authorities are ambiguous with a port separator.
        return "", None
    elif ":" in value:
        hostname, port_text = value.rsplit(":", 1)
        if not hostname or not port_text.isdigit():
            return "", None
    else:
        hostname = value
    port = int(port_text) if port_text else None
    if port is not None and not 0 < port < 65536:
        return "", None
    return hostname.lower(), port


def parse_public_origin(url: str) -> Optional[PublicOrigin]:
    """:class:`PublicOrigin` for a normalised public URL, or ``None`` when it has no usable
    http(s) host and port."""
    try:
        parsed = urlparse(url)
        hostname, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if parsed.scheme not in _DEFAULT_PORTS or not hostname:
        return None
    return PublicOrigin(parsed.scheme, hostname.lower(), port or _DEFAULT_PORTS[parsed.scheme], url)


def origins_from_urls(urls: Iterable[str]) -> Tuple[PublicOrigin, ...]:
    """Parsed origins in order, the first URL per origin winning."""
    seen: set = set()
    out = []
    for url in urls:
        origin = parse_public_origin(url)
        if origin is None:
            _log.warning("Public dashboard URL %r has no usable host or port; it is not served.", url)
            continue
        if origin.key not in seen:
            seen.add(origin.key)
            out.append(origin)
    return tuple(out)


def resolve_public_origins() -> Tuple[PublicOrigin, ...]:
    """Every configured public origin, primary first (empty when none is configured)."""
    return origins_from_urls(_prefix.resolve_public_urls())


def origin_header_key(value: str) -> Optional[OriginKey]:
    """``(scheme, host, port)`` of a web (http/https) ``Origin`` value, else ``None``."""
    try:
        parsed = urlparse((value or "").strip())
    except ValueError:
        return None
    if parsed.scheme not in _DEFAULT_PORTS or parsed.path or parsed.query or parsed.fragment:
        return None
    hostname, port = split_authority(parsed.netloc)
    if not hostname:
        return None
    return (parsed.scheme, hostname, port if port is not None else _DEFAULT_PORTS[parsed.scheme])


def classify_origin_header(value: str, origins: Sequence[PublicOrigin]) -> str:
    """``"non_web"`` (no Origin, ``null``, file://, app:// -- left to the credential check, as
    before), ``"listed"`` (exact match), ``"mismatch"`` (names a listed hostname with another
    scheme or port, or is malformed), or ``"unlisted"`` (some other web origin)."""
    raw = (value or "").strip()
    try:
        scheme = urlparse(raw).scheme if raw else ""
    except ValueError:
        return "mismatch"
    if scheme not in _DEFAULT_PORTS:
        return "non_web"
    key = origin_header_key(raw)
    if key is None:
        return "mismatch"
    if any(o.key == key for o in origins):
        return "listed"
    return "mismatch" if any(o.host == key[1] for o in origins) else "unlisted"


def forwarded_peer_trusted(scope) -> bool:
    """True only when :class:`ForwardedPeerMarker` saw a trusted proxy as the socket peer."""
    return scope.get(TRUSTED_PEER_SCOPE_KEY) is True


def _request_authority(conn) -> Optional[Tuple[str, str, Optional[int]]]:
    """``(scheme, host, explicit port or None)`` the request was addressed to, as the gateway
    can tell: the trusted proxy's ``X-Forwarded-Host`` or else the ``Host``, with the scheme
    uvicorn settled on (``X-Forwarded-Proto`` from the same trusted peers only).

    A trusted proxy must OVERWRITE ``X-Forwarded-Host`` with the host the browser asked for, not
    append to a value the client sent: only the first value is read. A proxy that appends lets
    the client choose that first value -- which can still only pick among LISTED origins, never
    put a host of its own into a redirect."""
    scheme = str(conn.scope.get("scheme") or "http")
    scheme = _WS_SCHEMES.get(scheme, scheme)
    if scheme not in _DEFAULT_PORTS:
        return None
    authority = conn.headers.get("host", "")
    if forwarded_peer_trusted(conn.scope):
        forwarded = conn.headers.get("x-forwarded-host", "").split(",")[0].strip()
        if forwarded:
            authority = forwarded
    hostname, port = split_authority(authority)
    return (scheme, hostname, port) if hostname else None


def request_origin_key(conn) -> Optional[OriginKey]:
    """The origin ``conn`` (a Starlette Request or WebSocket) was addressed to; a missing port
    is the scheme's default."""
    authority = _request_authority(conn)
    if authority is None:
        return None
    scheme, hostname, port = authority
    return (scheme, hostname, port if port is not None else _DEFAULT_PORTS[scheme])


def listed_origin_for(conn, origins: Sequence[PublicOrigin]) -> Optional[PublicOrigin]:
    """The listed origin ``conn`` was addressed to, or ``None``. A Host without a port (nginx
    ``$host`` drops it) also matches the one listed origin with that scheme and host when
    exactly one exists, whatever its port."""
    authority = _request_authority(conn)
    if authority is None:
        return None
    scheme, hostname, port = authority
    key = (scheme, hostname, port if port is not None else _DEFAULT_PORTS[scheme])
    exact = next((o for o in origins if o.key == key), None)
    if exact is not None or port is not None:
        return exact
    same = [o for o in origins if (o.scheme, o.host) == (scheme, hostname)]
    return same[0] if len(same) == 1 else None


def configured_origins(conn) -> Tuple[PublicOrigin, ...]:
    """The origins the running dashboard was started with (``app.state.public_origins``, the
    list the Host/Origin guards hold requests to), or the live config when no server set one
    (an app driven without ``start_server``)."""
    try:
        snapshot = getattr(conn.app.state, "public_origins", None)
    except (AttributeError, KeyError):
        snapshot = None
    return tuple(snapshot) if snapshot is not None else resolve_public_origins()


def public_base_url(conn) -> str:
    """Base URL for an absolute link back to the gateway (the OIDC and MCP OAuth callbacks):
    the listed URL of the origin ``conn`` came in on, else the primary. ``""`` when no public
    URL is configured (the caller reconstructs from the request, as upstream does)."""
    origins = configured_origins(conn)
    if not origins:
        return ""
    match = listed_origin_for(conn, origins)
    if match is not None:
        return match.base_url
    if len(origins) > 1:
        _warn_unmatched(request_origin_key(conn), origins)
    return origins[0].base_url


def _warn_unmatched(key: Optional[OriginKey], origins: Sequence[PublicOrigin]) -> None:
    """Say once why a request on a listed HOSTNAME fell back to the primary: the usual cause is
    a TLS proxy that is not in dashboard.trusted_proxies, so the gateway sees http."""
    if key is None or key in _warned_unmatched or not any(o.host == key[1] for o in origins):
        return
    _warned_unmatched.add(key)
    _log.warning(
        "Dashboard request for %s://%s:%d matches no listed public origin, so its sign-in uses "
        "the primary callback %s/auth/callback. If the browser used https, add the TLS proxy to "
        "dashboard.trusted_proxies so its X-Forwarded-Proto is honoured.",
        key[0], key[1], key[2], origins[0].base_url)


_warned_refused: set = set()
_WARNED_REFUSED_CAP = 256  # an attacker can send endless distinct Origins; never grow unbounded


def describe_origins(origins: Sequence[PublicOrigin]) -> str:
    return ", ".join(o.serialize() for o in origins) or "(none configured)"


def warn_refused_origin(kind: str, origin: str, origins: Sequence[PublicOrigin], fix: str) -> None:
    """Log once per (kind, Origin) that a browser Origin was refused, naming what is listed."""
    shown = (origin or "")[:200]
    if (kind, shown) in _warned_refused:
        return
    if len(_warned_refused) >= _WARNED_REFUSED_CAP:
        _warned_refused.clear()
    _warned_refused.add((kind, shown))
    _log.warning("Dashboard %s refused: Origin %r is not one of the listed origins (%s). %s",
                 kind, shown, describe_origins(origins), fix)


_WRITE_CHECK_MODES = {"auto": None, "on": True, "true": True, "off": False, "false": False}


def write_origin_check_enabled(origins: Sequence[PublicOrigin]) -> bool:
    """``dashboard.write_origin_check``: ``auto`` (default) is on with two or more public origins
    -- a single ``public_url`` gateway keeps upstream's behaviour -- ``on``/``off`` force it. It
    never runs without a declared public origin."""
    raw = _prefix._load_dashboard_section().get("write_origin_check", "auto")
    mode = str(raw).strip().lower() if raw is not None else "auto"
    if mode not in _WRITE_CHECK_MODES:
        _log.warning("dashboard.write_origin_check must be auto, on or off; %r read as auto", raw)
        mode = "auto"
    forced = _WRITE_CHECK_MODES[mode]
    enabled = len(origins) > 1 if forced is None else forced
    return bool(enabled and origins)


def callback_urls(origins: Sequence[PublicOrigin]) -> list[str]:
    """Every ``/auth/callback`` the OIDC client must have registered."""
    return [f"{o.base_url}/auth/callback" for o in origins]


class ForwardedPeerMarker:
    """Outermost ASGI wrapper: records whether the RAW socket peer is a trusted proxy before
    uvicorn's ``ProxyHeadersMiddleware`` replaces ``scope["client"]`` with the forwarded one."""

    def __init__(self, app, trusted: Sequence[str]):
        self.app = app
        # uvicorn trusts every peer for exactly "*" / ["*"] (config never lets that through).
        self._always = list(trusted) == ["*"]
        self._networks = []
        for entry in trusted:
            try:
                self._networks.append(ipaddress.ip_network(str(entry).strip(), strict=False))
            except ValueError:
                continue

    def trusts(self, peer: Optional[str]) -> bool:
        if self._always and peer:
            return True
        try:
            address = ipaddress.ip_address(peer or "")
        except ValueError:
            return False
        return any(address in network for network in self._networks)

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            client = scope.get("client")
            scope[TRUSTED_PEER_SCOPE_KEY] = self.trusts(client[0] if client else None)
        await self.app(scope, receive, send)


def install_forwarded_peer_marker(config) -> None:
    """Wrap a uvicorn ``Config``'s loaded app in :class:`ForwardedPeerMarker` when it trusts
    proxy headers, using the very list uvicorn trusts. A no-op otherwise. If uvicorn's internals
    no longer allow it, warn: ``X-Forwarded-Host`` is then never honoured (requests fall back to
    their ``Host`` and the primary origin), which is safe but not what the operator configured."""
    if not getattr(config, "proxy_headers", False):
        return
    try:
        if not getattr(config, "loaded", False):
            config.load()
        loaded = getattr(config, "loaded_app", None)
    except Exception as exc:  # noqa: BLE001 -- uvicorn reports the same failure when it loads
        loaded, reason = None, f"loading the app failed: {exc}"
    else:
        reason = "uvicorn's Config has no loaded_app"
    if loaded is None:
        _log.warning("Could not install the trusted-proxy marker (%s); X-Forwarded-Host will be "
                     "ignored and every sign-in uses its request Host or the primary origin.", reason)
        return
    trusted = getattr(config, "forwarded_allow_ips", None) or []
    if isinstance(trusted, str):
        trusted = [part.strip() for part in trusted.split(",")]
    config.loaded_app = ForwardedPeerMarker(loaded, list(trusted))
