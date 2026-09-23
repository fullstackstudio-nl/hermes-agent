"""The profile picture an identity provider sent at sign-in: fetched once, stored and served by
the gateway itself.

Why not hand the provider's URL to the app: every device that loads it tells the provider whose
conversation it is looking at, and such URLs expire. So the gateway fetches the image once, at
login, and serves its own copy behind the auth gate to the signed-in users of this gateway -- the
colleague whose name an author stamp shows can see that person's picture by the same id.

The URL comes out of an ID token, so it is untrusted input for SSRF purposes. The rules are
unconditional -- deliberately not ``tools.url_safety``, which honours the operator's
``security.allow_private_urls`` switch and proxy environment for the agent's own browsing:

* ``https`` on port 443 only, no credentials in the URL, an IDN host encoded to ASCII, at most
  :data:`MAX_REDIRECTS` redirects, and every hop is checked again.
* The name is resolved and EVERY answer must pass :func:`_refusal` (explicit deny-lists, so the
  verdict does not move with the Python patch release); the connection is then made to one of
  those vetted addresses, at most :data:`MAX_ADDRESSES` of them, while Host and SNI keep the name.
  NAT64 is looked through only for the well-known prefix ``64:ff9b::/96``; an operator-specific
  NAT64 prefix is not recognised (``tools.url_safety`` has the same gap).
* One deadline (:data:`_DEADLINE_SEC`, counted from when the login hands the fetch over) bounds all
  of it: the lookup, every connect, the TLS handshake, and every read and write, headers included.
* Only PNG, JPEG, WebP or GIF: the response must declare one AND its bytes must be one, with both
  dimensions at most :data:`MAX_DIMENSION` read from the file header without decoding. The type
  served later is the one read from the bytes.
* At most :data:`MAX_PICTURE_BYTES`, enforced while streaming.

Any failure means "no picture". A login never waits for a worker: see :func:`store_login_picture`.
"""
from __future__ import annotations

import hashlib
import ipaddress
import itertools
import logging
import os
import re
import socket
import tempfile
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import anyio
import anyio.to_thread
import httpcore
import httpx
import idna

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

MAX_PICTURE_BYTES = 512 * 1024
MAX_DIMENSION = 4096
MAX_REDIRECTS = 3
#: Answers dialled per name, the same cap as ``tools.url_safety``; all of them are still vetted.
MAX_ADDRESSES = 8
_CONNECT_TIMEOUT_SEC = 3.0
_READ_TIMEOUT_SEC = 5.0
_DEADLINE_SEC = 10.0
#: How long a login waits for its picture before going ahead without waiting any longer.
LOGIN_WAIT_SEC = _DEADLINE_SEC + 2.0
#: Login-time fetches in flight at once, gateway-wide. One identity holds at most one of them.
LOGIN_SLOTS = 4
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ACCEPTED_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_REQUEST_HEADERS = [
    (b"Accept", ", ".join(sorted(_ACCEPTED_TYPES)).encode("ascii")),
    (b"User-Agent", b"HermesAgent/1.0")]
_PATH_SAFE = "/%!$&'()*+,;=:@-._~"
#: An ASCII host is passed through as it is (underscores included, which IDNA refuses) but may
#: hold nothing else; it is still vetted when it is dialled.
_ASCII_HOST = re.compile(r"[a-z0-9._-]+\.?")

#: The endpoint :func:`picture_path` points at (``routes.api_auth_picture``).
PICTURE_ENDPOINT = "/api/auth/picture"

# Login-time fetches run on anyio threads under a limiter of their own, so the dashboard's shared
# threadpool tokens are never lent to a picture host. The limiter never makes anyone wait: a slot
# is reserved in ``_in_flight`` before the job is handed over, and a login finding no free slot
# skips its fetch instead.
_LOGIN_LIMITER = anyio.CapacityLimiter(LOGIN_SLOTS)
_login_lock = threading.Lock()
_in_flight: set[str] = set()  # identities with a fetch running; its size is the slots in use
_latest_generation: dict[str, int] = {}  # identity -> generation of its most recent sign-in
_generations = itertools.count(1)


class PictureRefused(Exception):
    """No picture from this URL: refused by the rules above, or the fetch failed."""


def identity_id(provider: str, user_id: str) -> str:
    """``<provider>:<user id>`` -- the same id the per-message author stamp carries, so a client can
    ask for the picture of whoever wrote a row."""
    return f"{provider.strip()}:{user_id.strip()}"


def picture_path(identity: str) -> str:
    """The gateway path serving ``identity``'s stored picture, relative to the gateway like every
    other ``/api`` path."""
    return f"{PICTURE_ENDPOINT}?id={quote(identity, safe='')}"


# ---- What an image is -----------------------------------------------------


def sniff_image_type(data: bytes) -> Optional[str]:
    """The accepted image type ``data`` actually is, from its magic bytes, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def _jpeg_size(data: bytes) -> Optional[tuple[int, int]]:
    """Width and height from the first start-of-frame segment, walking segment lengths only."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers carry no length
            i += 2
            continue
        if marker in (0xD8, 0xD9, 0xDA):  # a second SOI, EOI or scan data before any frame header
            return None
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2:
            return None
        if marker in _JPEG_SOF:
            if i + 9 > len(data):
                return None
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        i += 2 + length
    return None


def _webp_size(data: bytes) -> Optional[tuple[int, int]]:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        return (int.from_bytes(data[26:28], "little") & 0x3FFF,
                int.from_bytes(data[28:30], "little") & 0x3FFF)
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    return None


def _image_size(kind: str, data: bytes) -> Optional[tuple[int, int]]:
    """``(width, height)`` from the file header without decoding, else None when unreadable."""
    if kind == "image/png":
        if len(data) < 24 or data[12:16] != b"IHDR":
            return None
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if kind == "image/gif":
        if len(data) < 10:
            return None
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if kind == "image/jpeg":
        return _jpeg_size(data)
    return _webp_size(data)


def _checked_image(data: bytes) -> str:
    """The type ``data`` is, provided it is an accepted image no larger than the dimension cap."""
    kind = sniff_image_type(data)
    if kind is None:
        raise PictureRefused("picture bytes are not PNG, JPEG, WebP or GIF")
    size = _image_size(kind, data)
    if size is None or not all(size):
        raise PictureRefused(f"cannot read the size of this {kind}")
    if max(size) > MAX_DIMENSION:
        raise PictureRefused(f"picture is {size[0]}x{size[1]}, larger than the cap")
    return kind


# ---- Where a fetch may go -------------------------------------------------

_net = ipaddress.ip_network
#: Never dialled: private, loopback, link-local, shared, special-purpose, documentation,
#: benchmarking, multicast and reserved IPv4 space.
_V4_DENY = tuple(_net(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.0.0.0/24", "192.0.2.0/24", "192.31.196.0/24", "192.52.193.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "192.175.48.0/24", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4"))
#: IPv6 forms that stand for an IPv4 address in their low 32 bits: IPv4-compatible, IPv4-mapped,
#: IPv4-translated and NAT64. That IPv4 address is what gets vetted.
_V6_LOW32_V4 = tuple(_net(n) for n in ("::/96", "::ffff:0:0/96", "::ffff:0:0:0/96", "64:ff9b::/96"))
_V6_6TO4 = _net("2002::/16")
#: IPv6 outside global unicast is refused outright: link-local, site-local (fec0::/10), unique
#: local, multicast, discard-only, SRv6 SIDs and local-use NAT64 are all outside it.
_V6_GLOBAL_UNICAST = _net("2000::/3")
#: Inside global unicast but still never a picture host: the IETF protocol block (Teredo,
#: benchmarking, ORCHID, ...) and both documentation prefixes.
_V6_DENY = tuple(_net(n) for n in ("2001::/23", "2001:db8::/32", "3fff::/20"))


def _v4_refusal(ip: ipaddress.IPv4Address, shown: object) -> Optional[str]:
    if any(ip in n for n in _V4_DENY):
        return f"{shown} is not a public address"
    return None


def _refusal(address: str) -> Optional[str]:
    """Why ``address`` may not be dialled, else None."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return f"{address!r} is not an IP address"
    if isinstance(ip, ipaddress.IPv4Address):
        return _v4_refusal(ip, ip)
    if any(ip in n for n in _V6_LOW32_V4):
        return _v4_refusal(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF), address)
    if ip in _V6_6TO4:
        embedded = ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
        if (reason := _v4_refusal(embedded, address)) is not None:
            return reason
    if ip not in _V6_GLOBAL_UNICAST or any(ip in n for n in _V6_DENY):
        return f"{address} is not a public address"
    return None


def _resolve(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (a seam for tests)."""
    return [str(info[4][0]).split("%")[0]
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _network_backend() -> httpcore.NetworkBackend:
    """The real sockets underneath the guard (a seam for tests)."""
    return httpcore.SyncBackend()


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise PictureRefused("picture fetch ran out of time")
    return left


def _clamp(timeout: Optional[float], deadline: float) -> float:
    left = _remaining(deadline)
    return left if timeout is None else min(timeout, left)


def _lookup(host: str, port: int, deadline: float) -> list[str]:
    """``_resolve`` cut off at the deadline. ``getaddrinfo`` has no timeout of its own, so it runs on
    a daemon thread of its own: one that hangs is abandoned here and can never hold up the
    interpreter's exit, and so never a gateway restart."""
    future: Future = Future()

    def run() -> None:
        try:
            future.set_result(_resolve(host, port))
        except BaseException as exc:  # noqa: BLE001 — handed to the waiting fetch as-is
            future.set_exception(exc)

    threading.Thread(target=run, name="hermes-picture-dns", daemon=True).start()
    try:
        return future.result(timeout=_remaining(deadline))
    except FutureTimeout:
        raise PictureRefused(f"{host} did not resolve in time") from None
    except OSError as exc:
        raise PictureRefused(f"{host} does not resolve") from exc


def _vetted_addresses(host: str, port: int, deadline: float) -> list[str]:
    try:
        ipaddress.ip_address(host)
        answers = [host]
    except ValueError:
        answers = _lookup(host, port, deadline)
    if not answers:
        raise PictureRefused(f"{host} does not resolve")
    for answer in answers:
        if (reason := _refusal(answer)) is not None:
            raise PictureRefused(f"{host}: {reason}")
    return list(dict.fromkeys(answers))[:MAX_ADDRESSES]


class _DeadlineStream(httpcore.NetworkStream):
    """Every read, write and TLS handshake gets at most the time left before the deadline, so a
    peer trickling one byte at a time cannot stretch the fetch past it."""

    def __init__(self, stream: httpcore.NetworkStream, deadline: float) -> None:
        self._stream = stream
        self._deadline = deadline

    def read(self, max_bytes, timeout=None):
        return self._stream.read(max_bytes, timeout=_clamp(timeout, self._deadline))

    def write(self, buffer, timeout=None):
        self._stream.write(buffer, timeout=_clamp(timeout, self._deadline))

    def close(self):
        self._stream.close()

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        tls = self._stream.start_tls(
            ssl_context, server_hostname=server_hostname, timeout=_clamp(timeout, self._deadline))
        return _DeadlineStream(tls, self._deadline)

    def get_extra_info(self, info):
        return self._stream.get_extra_info(info)


class _PinnedBackend(httpcore.NetworkBackend):
    """Resolves and vets at connect time, then dials a vetted address. httpcore keeps the URL's
    name for Host and TLS, so the certificate is still checked against the name."""

    def __init__(self, inner: httpcore.NetworkBackend, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last: Optional[Exception] = None
        for address in _vetted_addresses(host, port, self._deadline):
            try:
                stream = self._inner.connect_tcp(
                    address, port, timeout=_clamp(timeout, self._deadline),
                    local_address=local_address, socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last = exc
                continue
            return _DeadlineStream(stream, self._deadline)
        raise last or PictureRefused(f"{host} does not resolve")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise PictureRefused("unix sockets are never a picture source")

    def sleep(self, seconds):
        self._inner.sleep(seconds)


def _normalise_url(url: str) -> str:
    """``url`` as the one https URL the fetch will request, or :class:`PictureRefused`."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise PictureRefused("malformed picture URL") from exc
    if parts.scheme != "https":
        raise PictureRefused(f"picture URL must be https, not {parts.scheme or 'schemeless'}")
    if not host:
        raise PictureRefused("picture URL has no host")
    if parts.username is not None or parts.password is not None:
        raise PictureRefused("picture URL carries credentials")
    if port not in (None, 443):
        raise PictureRefused("picture URL must use port 443")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if host.isascii():
            if not _ASCII_HOST.fullmatch(host):
                raise PictureRefused("picture URL host is not a valid name") from None
            netloc = host
        else:
            try:
                netloc = idna.encode(host, uts46=True).decode("ascii")
            except (idna.IDNAError, UnicodeError) as exc:
                raise PictureRefused("picture URL host is not a valid name") from exc
    else:
        if (reason := _refusal(host)) is not None:
            raise PictureRefused(reason)
        netloc = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", netloc, quote(parts.path, safe=_PATH_SAFE) or "/",
                       quote(parts.query, safe=_PATH_SAFE + "?"), ""))


# ---- Fetch ----------------------------------------------------------------


def _get(pool: httpcore.ConnectionPool, url: str, deadline: float) -> tuple[Optional[str], bytes]:
    """One hop: ``(redirect location, b"")`` or ``(None, body)``."""
    timeouts = {"connect": _clamp(_CONNECT_TIMEOUT_SEC, deadline),
                "read": _clamp(_READ_TIMEOUT_SEC, deadline),
                "write": _clamp(_READ_TIMEOUT_SEC, deadline),
                "pool": _clamp(_CONNECT_TIMEOUT_SEC, deadline)}
    with pool.stream("GET", url, headers=_REQUEST_HEADERS, extensions={"timeout": timeouts}) as resp:
        headers = {name.lower(): value for name, value in resp.headers}
        if resp.status in _REDIRECT_STATUSES:
            location = headers.get(b"location")
            if not location:
                raise PictureRefused("redirect without a location")
            return location.decode("latin-1"), b""
        if resp.status != 200:
            raise PictureRefused(f"picture fetch answered {resp.status}")
        declared = headers.get(b"content-type", b"").split(b";")[0].strip().lower().decode("latin-1")
        if declared not in _ACCEPTED_TYPES:
            raise PictureRefused(f"picture declared as {declared or 'nothing'}")
        try:
            declared_length = int(headers.get(b"content-length", b"0"))
        except ValueError as exc:
            raise PictureRefused("malformed content-length") from exc
        if declared_length > MAX_PICTURE_BYTES:
            raise PictureRefused("picture is larger than the cap")
        body = bytearray()
        for chunk in resp.iter_stream():
            body += chunk
            if len(body) > MAX_PICTURE_BYTES:
                raise PictureRefused("picture is larger than the cap")
        return None, bytes(body)


def fetch_picture(url: str, deadline: Optional[float] = None) -> tuple[bytes, str]:
    """``(bytes, type read from the bytes)`` for the picture at ``url``, under the rules in the
    module docstring, by ``deadline`` (``time.monotonic()``; default :data:`_DEADLINE_SEC` from
    now). Raises :class:`PictureRefused` for every way of not getting one."""
    if deadline is None:
        deadline = time.monotonic() + _DEADLINE_SEC
    backend = _PinnedBackend(_network_backend(), deadline)
    with httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(), network_backend=backend, retries=0) as pool:
        for _hop in range(MAX_REDIRECTS + 1):
            url = _normalise_url(url)
            try:
                location, body = _get(pool, url, deadline)
            except (httpcore.TimeoutException, httpcore.NetworkError, httpcore.ProtocolError,
                    httpcore.UnsupportedProtocol) as exc:
                raise PictureRefused(f"picture fetch failed: {type(exc).__name__}") from exc
            if location is None:
                break
            url = urljoin(url, location)
        else:
            raise PictureRefused("too many redirects")
    return body, _checked_image(body)


# ---- Store ----------------------------------------------------------------


def _picture_file(identity: str) -> Path:
    """Keyed by a hash of the identity, so the file name neither spells out who it is nor can be
    steered outside the directory by an id."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return get_hermes_home() / "dashboard_auth" / "pictures" / digest


def _store(identity: str, data: Optional[bytes]) -> None:
    path = _picture_file(identity)
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, incoming = tempfile.mkstemp(dir=path.parent, prefix=".incoming-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(incoming, path)
    except BaseException:
        Path(incoming).unlink(missing_ok=True)
        raise


def _store_if_latest(identity: str, generation: int, data: Optional[bytes]) -> bool:
    """Store (or, for ``None``, remove) the picture only while ``generation`` is still the latest
    sign-in of ``identity``: the last login wins, not the last fetch to finish. Checked and written
    under the lock that hands out generations, so no newer sign-in can slip in between."""
    with _login_lock:
        if _latest_generation.get(identity) != generation:
            return False
        _store(identity, data)
    return data is not None


def _fetch_and_store(session, identity: str, generation: int, deadline: float) -> None:
    """The picture this sign-in brought, or none when it brought none or its fetch failed. Never
    raises: a sign-in must not fail over a picture."""
    data: Optional[bytes] = None
    if session.picture:
        try:
            data, _kind = fetch_picture(session.picture, deadline)
        except Exception as exc:  # noqa: BLE001 — whatever went wrong, the login goes ahead
            logger.info("dashboard-auth: no profile picture stored at sign-in: %s", exc)
    try:
        _store_if_latest(identity, generation, data)
    except OSError as exc:
        logger.warning("dashboard-auth: could not update the stored profile picture: %s", exc)


def _claim(identity: str) -> Optional[int]:
    """Record a sign-in of ``identity`` and reserve a fetch slot for it: its generation, or None
    when this sign-in must not fetch -- the identity already has a fetch running, or every slot is
    taken. Either way the sign-in is now the latest, so an older fetch still running can no longer
    store over what is kept."""
    with _login_lock:
        generation = next(_generations)
        _latest_generation[identity] = generation
        if identity in _in_flight or len(_in_flight) >= LOGIN_SLOTS:
            return None
        _in_flight.add(identity)
        return generation


class _LoginFetch:
    """One reserved slot's job. The slot is given back when the job ends -- or, if the login stops
    waiting before a thread ever picked the job up, by the login, and the job then never runs."""

    def __init__(self, session, identity: str, generation: int, deadline: float) -> None:
        self._session, self._identity = session, identity
        self._generation, self._deadline = generation, deadline
        self._state = "queued"

    def run(self) -> None:
        with _login_lock:
            if self._state == "abandoned":
                return
            self._state = "running"
        try:
            _fetch_and_store(self._session, self._identity, self._generation, self._deadline)
        finally:
            with _login_lock:
                _in_flight.discard(self._identity)

    def abandon_if_queued(self) -> None:
        with _login_lock:
            if self._state == "queued":
                self._state = "abandoned"
                _in_flight.discard(self._identity)


async def store_login_picture(session) -> None:
    """Refresh the stored picture for a sign-in, waiting at most :data:`LOGIN_WAIT_SEC`.

    A login never waits for a worker. When this identity already has a fetch running, or all
    :data:`LOGIN_SLOTS` are taken, the fetch is skipped and the stored picture is kept as it is --
    a busy gateway never deletes anyone's picture; the next sign-in refreshes it. Otherwise the
    fetch starts at once with its deadline counted from now, and the login goes ahead when it ends
    or when the wait runs out, whatever the fetch thread is still doing."""
    identity = identity_id(session.provider, session.user_id)
    generation = _claim(identity)
    if generation is None:
        logger.info("dashboard-auth: profile picture not refreshed at this sign-in (fetch busy)")
        return
    job = _LoginFetch(session, identity, generation, time.monotonic() + _DEADLINE_SEC)
    try:
        with anyio.move_on_after(LOGIN_WAIT_SEC):
            await anyio.to_thread.run_sync(job.run, limiter=_LOGIN_LIMITER, abandon_on_cancel=True)
    finally:
        job.abandon_if_queued()


def has_picture(identity: str) -> bool:
    return bool(identity) and _picture_file(identity).is_file()


def read_picture(identity: str) -> Optional[tuple[bytes, str]]:
    """``(bytes, type read from the bytes)`` stored for ``identity``, else None."""
    if not identity:
        return None
    try:
        data = _picture_file(identity).read_bytes()
    except OSError:
        return None
    kind = sniff_image_type(data)
    return (data, kind) if kind else None
