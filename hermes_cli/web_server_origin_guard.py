"""``Origin`` checks for the dashboard: WebSocket upgrades and cookie-authenticated writes.

Fork-only (fullstackstudio-org/hermes-agent), built on ``dashboard_auth.origins``: with several
public origins listed, each is accepted exactly (scheme + host + port) and nothing else is
loosened. The Host guard itself stays in ``web_server._is_accepted_host``; this module reaches
it (and ``app.state``) late, like ``web_server_chat`` does, so importing it never imports the
app.
"""
from __future__ import annotations

import urllib.parse
from typing import Optional

from fastapi.responses import JSONResponse

from hermes_cli.dashboard_auth.origins import (
    classify_origin_header, describe_origins, origin_header_key, split_authority, warn_refused_origin)
from hermes_cli.dashboard_auth.request_utils import extract_bearer

_WILDCARD_BINDS = frozenset({"0.0.0.0", "::"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
WS_ORIGIN_FIX = ("Open the dashboard on a listed origin, or list this one in dashboard.public_url / "
                 "dashboard.public_urls with its exact scheme and port.")
_WRITE_ORIGIN_FIX = ("If the dashboard is opened there, add it to dashboard.public_urls; to switch this "
                     "check off, set dashboard.write_origin_check: off.")


def _is_accepted_origin(
    origin: str,
    bound_host: str,
    trusted_public_hosts: frozenset[str] = frozenset(),
    public_origins: tuple = (),
    *,
    host_header: str = "",
    wildcard_bind_accepts: bool = True,
) -> bool:
    """True if a request's ``Origin`` may drive this dashboard.

    Non-web origins (none, ``null``, packaged Electron file:// / app://) are left to the
    credential check, as before. A listed public origin is accepted exactly -- scheme, host and
    port. A variant of a listed hostname on another scheme or port is refused unless it is the
    request's own explicit ``host:port`` (one socket serves one scheme, so that is the page's own
    origin: e.g. the dashboard opened on ``:9443`` of a ``public_url`` written without a port).
    Any other web origin must be same-origin with the request's ``Host`` or pass the bound-host
    rules; with ``wildcard_bind_accepts=False`` a 0.0.0.0 bind no longer accepts every origin.
    """
    from hermes_cli.web_server import _is_accepted_host

    verdict = classify_origin_header(origin, public_origins)
    if verdict in ("non_web", "listed"):
        return True
    key = origin_header_key(origin)
    request_host, request_port = split_authority(host_header)
    if verdict == "mismatch":
        return bool(key and request_port is not None and (key[1], key[2]) == (request_host, request_port))
    if key and key[1] == request_host and request_port in (None, key[2]):
        return True
    if not wildcard_bind_accepts and bound_host in _WILDCARD_BINDS:
        return False
    netloc = urllib.parse.urlparse(origin.strip()).netloc
    return _is_accepted_host(netloc, bound_host, trusted_public_hosts)


def _cross_origin_write_refusal(request) -> Optional[JSONResponse]:
    """403 for a cookie-authenticated write whose browser ``Origin`` is neither a listed origin
    nor the request's own, while ``dashboard.write_origin_check`` is on (by default: two or more
    public origins); else ``None``.

    ``SameSite=Lax`` does not cover this: every ``*.example.com`` is the same SITE, so a page on
    a sibling subdomain would otherwise get the session cookie attached to its POST. Bearer
    callers are exempt (a browser never attaches one on its own, and the gate never falls back
    to cookies when one is sent), as are requests without an ``Origin`` (non-browser clients)
    and non-web origins (packaged desktop)."""
    from hermes_cli.web_server import app

    public_origins = getattr(app.state, "public_origins", ())
    if (request.method in _SAFE_METHODS or not public_origins
            or not getattr(app.state, "write_origin_check", False)
            or not getattr(app.state, "auth_required", False)
            or extract_bearer(request)):
        return None
    origin = request.headers.get("origin", "")
    if _is_accepted_origin(
            origin, getattr(app.state, "bound_host", None) or "",
            getattr(app.state, "trusted_public_hosts", frozenset()), public_origins,
            host_header=request.headers.get("host", ""), wildcard_bind_accepts=False):
        return None
    warn_refused_origin("write request", origin, public_origins, _WRITE_ORIGIN_FIX)
    return JSONResponse(status_code=403, content={
        "detail": (f"Cross-origin request refused: Origin {origin[:200]!r} is not one of this "
                   f"dashboard's origins ({describe_origins(public_origins)}). {_WRITE_ORIGIN_FIX}"),
        "reason": "origin_not_listed"})
