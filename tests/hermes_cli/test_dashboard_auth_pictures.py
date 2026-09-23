"""The signed-in user's email and profile picture, as the identity provider sent them.

Two halves:

* ``fetch_picture`` — the picture URL comes out of an ID token, so it is untrusted input. Only
  ``https``, never a loopback / private / link-local answer (checked at connect time, on every
  redirect hop, and the vetted IP is the one dialled), at most a few redirects, a 512 KB cap while
  streaming, and the bytes themselves must be PNG / JPEG / WebP / GIF.
* The real login round trip through the self-hosted OIDC provider and the gated dashboard app:
  the picture is fetched once at login, stored by identity, replaced on every login, served only
  to signed-in users, and a failed fetch never fails the login.

The network below the HTTP client is replaced by an in-process upstream (``_Upstream``) and the
resolver by a table, so the real guard, the real HTTP parsing and the real routes all run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
import time
import urllib.parse
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import anyio
import anyio.to_thread
import httpcore
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import plugins.dashboard_auth.self_hosted as oidc_plugin
from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, native_flow, pictures, register_provider

_ISSUER = "https://id.example.org/realms/team"
_CLIENT_ID = "hermes-dashboard"
_AVATAR_HOST = "avatars.example.org"
_AVATAR_URL = f"https://{_AVATAR_HOST}/u/sam.png"
_PUBLIC_IP = "93.184.215.14"  # example.com's address; never dialled, the upstream is in-process



# ---- Minimal images: real headers (the size is read from them), no pixel data needed ----

def png(w: int = 64, h: int = 64, pad: bytes = b"\x00" * 16) -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + w.to_bytes(4, "big")
            + h.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00" + pad)


def jpeg(w: int = 64, h: int = 64) -> bytes:
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    sof0 = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + h.to_bytes(2, "big")
            + w.to_bytes(2, "big") + b"\x03" + b"\x00" * 9)
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def gif(w: int = 64, h: int = 64) -> bytes:
    return b"GIF89a" + w.to_bytes(2, "little") + h.to_bytes(2, "little") + b"\x00" * 8


def _riff(chunk: bytes) -> bytes:
    return b"RIFF" + (len(chunk) + 4).to_bytes(4, "little") + b"WEBP" + chunk


def webp_vp8(w: int = 64, h: int = 64) -> bytes:
    frame = b"\x00\x00\x00" + b"\x9d\x01\x2a" + w.to_bytes(2, "little") + h.to_bytes(2, "little")
    return _riff(b"VP8 " + (len(frame) + 8).to_bytes(4, "little") + frame + b"\x00" * 8)


def webp_vp8l(w: int = 64, h: int = 64) -> bytes:
    bits = (w - 1) | ((h - 1) << 14)
    return _riff(b"VP8L" + (13).to_bytes(4, "little") + b"\x2f" + bits.to_bytes(4, "little")
                 + b"\x00" * 8)


def webp_vp8x(w: int = 64, h: int = 64) -> bytes:
    return _riff(b"VP8X" + (10).to_bytes(4, "little") + b"\x00" * 4
                 + (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little") + b"\x00" * 8)


PNG = png()
PNG_B = png(pad=b"\x01" * 16)
JPEG = jpeg()
GIF = gif()
WEBP = webp_vp8()


# ---------------------------------------------------------------------------
# In-process upstream + resolver
# ---------------------------------------------------------------------------


def _response(status: int = 200, body: bytes = b"", *, ctype: str | None = "image/png",
              length: bool = True, extra: tuple[str, ...] = ()) -> list[bytes]:
    head = [f"HTTP/1.1 {status} X", "Connection: close"]
    if ctype:
        head.append(f"Content-Type: {ctype}")
    if length:
        head.append(f"Content-Length: {len(body)}")
    head.extend(extra)
    return [("\r\n".join(head) + "\r\n\r\n").encode(), *([body] if body else [])]


class _Upstream(httpcore.NetworkBackend):
    """One canned response per TCP connection, recording every address actually dialled."""

    def __init__(self, *responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.dialed: list[tuple[str, int]] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.dialed.append((host, port))
        if not self.responses:
            raise httpcore.ConnectError("upstream unreachable")
        return httpcore.MockStream(list(self.responses.pop(0)))

    def connect_unix_socket(self, path, timeout=None, socket_options=None):  # pragma: no cover
        raise AssertionError("never a unix socket")

    def sleep(self, seconds):  # pragma: no cover
        pass


@pytest.fixture
def net(monkeypatch):
    """Install a resolver table and an upstream; returns a setter for both."""
    state: Dict[str, Any] = {"table": {_AVATAR_HOST: [_PUBLIC_IP]}, "upstream": _Upstream()}

    def resolve(host: str, port: int) -> list[str]:
        try:
            return list(state["table"][host])
        except KeyError:
            raise OSError(f"no such host: {host}") from None

    monkeypatch.setattr(pictures, "_resolve", resolve)
    monkeypatch.setattr(pictures, "_network_backend", lambda: state["upstream"])

    class _Net:
        table = state["table"]

        @staticmethod
        def serve(*responses: list[bytes]) -> _Upstream:
            state["upstream"] = _Upstream(*responses)
            return state["upstream"]

    return _Net


# ---------------------------------------------------------------------------
# fetch_picture: the SSRF / type / size rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body,kind", [
    (PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp"),
    (webp_vp8l(), "image/webp"), (webp_vp8x(), "image/webp")])
def test_fetch_accepts_the_four_image_types_and_dials_the_vetted_ip(net, body, kind):
    upstream = net.serve(_response(body=body, ctype=kind))
    assert pictures.fetch_picture(_AVATAR_URL) == (body, kind)
    # The connection went to the address the guard vetted, not to a second lookup of the name.
    assert upstream.dialed == [(_PUBLIC_IP, 443)]


def test_served_type_comes_from_the_bytes_not_the_header(net):
    net.serve(_response(body=PNG, ctype="image/jpeg"))
    assert pictures.fetch_picture(_AVATAR_URL) == (PNG, "image/png")


@pytest.mark.parametrize("url", [
    "http://avatars.example.org/u/sam.png",       # not https
    "ftp://avatars.example.org/u/sam.png",
    "https://127.0.0.1/sam.png",                  # loopback literal
    "https://[::1]/sam.png",
    "https://10.0.0.7/sam.png",                   # private literal
    "https://192.168.1.4/sam.png",
    "https://169.254.169.254/latest/meta-data",   # link-local / metadata
    "https://[::ffff:127.0.0.1]/sam.png",         # v4-mapped loopback
    "https://user:pw@avatars.example.org/sam.png",
    "https://avatars.example.org:8443/sam.png",   # only port 443
    "https://avatars.example.org:80/sam.png",
    "https://[fe80::1%25lo0]/sam.png",
    "not a url",
])
def test_fetch_refuses_urls_that_are_not_public_https(net, url):
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(url)
    assert upstream.dialed == []


@pytest.mark.parametrize("answer", [
    ["127.0.0.1"], ["10.1.2.3"], ["172.16.0.9"], ["192.168.0.2"], ["169.254.1.1"], ["fe80::1"],
    ["::1"], ["100.64.0.1"], ["0.0.0.0"],
    # One bad answer among good ones is enough to refuse the name.
    [_PUBLIC_IP, "10.1.2.3"]])
def test_fetch_refuses_a_name_that_resolves_to_a_private_address(net, answer):
    net.table[_AVATAR_HOST] = answer
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert upstream.dialed == []


def test_an_explicit_port_443_is_fine(net):
    upstream = net.serve(_response(body=PNG))
    assert pictures.fetch_picture(f"https://{_AVATAR_HOST}:443/u/sam.png") == (PNG, "image/png")
    assert upstream.dialed == [(_PUBLIC_IP, 443)]


def test_an_ascii_host_with_an_underscore_is_passed_through(net):
    net.table["my_bucket.storage.example.org"] = [_PUBLIC_IP]
    net.serve(_response(body=PNG))
    assert pictures.fetch_picture("https://my_bucket.storage.example.org/x.png") == (PNG, "image/png")


@pytest.mark.parametrize("url", ["https://a b.example.org/x.png", "https://a*b.example.org/x.png"])
def test_an_ascii_host_with_anything_else_is_refused(net, url):
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(url)
    assert upstream.dialed == []


def test_an_idn_host_is_encoded_before_it_is_looked_up(net):
    net.table["xn--bld-zma.example.org"] = [_PUBLIC_IP]
    net.serve(_response(body=PNG))
    assert pictures.fetch_picture("https://bïld.example.org/ü.png") == (PNG, "image/png")


def test_a_lookalike_of_localhost_is_just_a_name_that_does_not_resolve(net):
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture("https://l\u043ecalhost/x.png")  # Cyrillic o
    assert upstream.dialed == []


# Every IPv6 form that carries an IPv4 address, each wrapping a private and a loopback one.
_WRAPPED_PRIVATE = {
    "mapped": ["::ffff:10.0.0.1", "::ffff:127.0.0.1"],
    "compatible": ["::10.0.0.1", "::7f00:1"],
    "translated": ["::ffff:0:10.0.0.1", "::ffff:0:7f00:1"],
    "6to4": ["2002:a00:1::1", "2002:7f00:1::"],
    "teredo": ["2001:0:4136:e378:8000:63bf:f5ff:fffe", "2001:0:4136:e378:8000:63bf:80ff:fffe"],
    "nat64": ["64:ff9b::10.0.0.1", "64:ff9b::127.0.0.1"],
    "local-use nat64": ["64:ff9b:1::a00:1"],
}


@pytest.mark.parametrize("answer", [
    pytest.param(a, id=f"{form}-{a}") for form, addrs in _WRAPPED_PRIVATE.items() for a in addrs])
def test_every_ipv6_form_wrapping_a_private_ipv4_is_refused(net, answer):
    net.table[_AVATAR_HOST] = [answer]
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert upstream.dialed == []


@pytest.mark.parametrize("answer", [
    "::ffff:93.184.215.14", "64:ff9b::93.184.215.14", "2002:5db8:d70e::1"])
def test_a_wrapped_public_ipv4_is_vetted_as_that_address(net, answer):
    net.table[_AVATAR_HOST] = [answer]
    net.serve(_response(body=PNG))
    assert pictures.fetch_picture(_AVATAR_URL) == (PNG, "image/png")


@pytest.mark.parametrize("answer", [
    # multicast
    "224.0.0.1", "239.255.255.250", "ff02::1", "ff0e::1",
    # reserved, special-purpose, documentation, benchmarking
    "240.0.0.1", "255.255.255.255", "0.1.2.3", "192.0.0.8", "192.0.2.1", "192.88.99.1",
    "198.18.0.1", "198.51.100.1", "fec0::1", "fc00::1", "100::1", "5f00::1", "3fff::1",
    "2001:db8::1", "2001:2::1",
    # Teredo is refused whatever it wraps
    "2001:0:4136:e378:8000:63bf:a247:28f1"])
def test_multicast_and_reserved_addresses_are_refused(net, answer):
    net.table[_AVATAR_HOST] = [answer]
    upstream = net.serve(_response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert upstream.dialed == []


def test_every_redirect_hop_is_checked_again(net):
    net.table["internal.example.org"] = ["10.9.9.9"]
    upstream = net.serve(
        _response(302, ctype=None, extra=("Location: https://internal.example.org/x.png",)),
        _response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert upstream.dialed == [(_PUBLIC_IP, 443)]


def test_a_redirect_to_plain_http_is_refused(net):
    upstream = net.serve(
        _response(302, ctype=None, extra=(f"Location: http://{_AVATAR_HOST}/x.png",)),
        _response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert len(upstream.dialed) == 1


def test_redirects_are_followed_up_to_the_cap_and_no_further(net):
    hop = _response(302, ctype=None, extra=(f"Location: {_AVATAR_URL}",))
    net.serve(*([hop] * pictures.MAX_REDIRECTS), _response(body=PNG))
    assert pictures.fetch_picture(_AVATAR_URL) == (PNG, "image/png")

    upstream = net.serve(*([hop] * (pictures.MAX_REDIRECTS + 1)), _response(body=PNG))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)
    assert len(upstream.dialed) == pictures.MAX_REDIRECTS + 1


@pytest.mark.parametrize("response", [
    _response(body=b"<html>not a picture</html>", ctype="image/png"),   # header lies
    _response(body=PNG, ctype="text/html"),                             # header refuses
    _response(body=PNG, ctype=None),                                    # no header at all
    _response(body=b"<svg xmlns='http://www.w3.org/2000/svg'/>", ctype="image/svg+xml"),
    _response(404, body=PNG),
])
def test_fetch_refuses_anything_that_is_not_an_accepted_image(net, response):
    net.serve(response)
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)


def test_fetch_refuses_an_oversized_body_by_its_declared_length(net):
    net.serve(_response(body=PNG + b"\x00" * pictures.MAX_PICTURE_BYTES))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)


def test_fetch_refuses_an_oversized_body_while_streaming_without_a_length(net):
    chunk = b"\x00" * 65536
    head, *_ = _response(ctype="image/png", length=False)
    net.serve([head, PNG, *([chunk] * 9)])
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)


@pytest.mark.parametrize("body", [
    png(4097, 16), png(16, 4097), jpeg(4097, 16), jpeg(16, 4097), gif(4097, 16), gif(16, 4097),
    webp_vp8(4097, 16), webp_vp8l(16, 4097), webp_vp8x(4097, 16)])
def test_a_picture_larger_than_the_dimension_cap_is_refused(net, body):
    net.serve(_response(body=body))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)


@pytest.mark.parametrize("body", [
    png(4096, 4096), jpeg(4096, 4096), gif(4096, 4096), webp_vp8x(4096, 4096)])
def test_a_picture_at_the_dimension_cap_is_accepted(net, body):
    net.serve(_response(body=body))
    assert pictures.fetch_picture(_AVATAR_URL)[0] == body


@pytest.mark.parametrize("body", [
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,                    # no IHDR
    png(0, 16),                                               # zero width
    b"\xff\xd8\xff\xda" + b"\x00" * 32,                       # scan data before any frame header
    b"\xff\xd8\xff\xe0\x00\x10" + b"\x00" * 8,                 # ends before a frame header
    _riff(b"ABCD" + b"\x00" * 20),                            # unknown WebP chunk
    b"GIF89a\x10",                                            # truncated screen descriptor
])
def test_a_picture_whose_size_cannot_be_read_is_refused(net, body):
    net.serve(_response(body=body))
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(_AVATAR_URL)


def test_a_body_exactly_at_the_cap_is_accepted(net):
    body = PNG + b"\x00" * (pictures.MAX_PICTURE_BYTES - len(PNG))
    net.serve(_response(body=body))
    assert pictures.fetch_picture(_AVATAR_URL) == (body, "image/png")


# ---------------------------------------------------------------------------
# One deadline bounds the whole fetch
# ---------------------------------------------------------------------------

#: Wall-clock bound for a 0.5 s deadline. Every case below takes far longer without it (18 s of
#: blackholed connects, ~10 s of trickled headers, a 3 s handshake, a lookup that never returns).
_BOUND = 2.5


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setattr(pictures, "_DEADLINE_SEC", 0.5)


class _Backend(httpcore.NetworkBackend):
    def __init__(self, connect) -> None:
        self._connect = connect
        self.dialed: list[tuple[str, float]] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.dialed.append((host, timeout))
        return self._connect(timeout)

    def sleep(self, seconds):  # pragma: no cover
        pass


class _SlowStream(httpcore.NetworkStream):
    """Answers one byte per read, each read well inside any per-read timeout; or stalls the TLS
    handshake for as long as it is allowed to."""

    def __init__(self, data: bytes = b"", *, stall_tls: bool = False,
                 stall_write: bool = False) -> None:
        self._data, self._i = data, 0
        self._stall_tls, self._stall_write = stall_tls, stall_write

    def read(self, max_bytes, timeout=None):
        time.sleep(0.05)
        chunk = self._data[self._i:self._i + 1]
        self._i += 1
        return chunk

    def write(self, buffer, timeout=None):
        if self._stall_write:
            time.sleep(timeout)

    def close(self):
        pass

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        if self._stall_tls:
            time.sleep(timeout)
            raise httpcore.ConnectTimeout("handshake stalled")
        return self

    def get_extra_info(self, info):
        return None


def _timed_refusal(url: str = _AVATAR_URL) -> float:
    started = time.monotonic()
    with pytest.raises(pictures.PictureRefused):
        pictures.fetch_picture(url)
    return time.monotonic() - started


def test_blackholed_addresses_cannot_stretch_the_fetch_past_the_deadline(
        net, monkeypatch, short_deadline):
    net.table[_AVATAR_HOST] = [f"93.184.215.{i}" for i in range(20, 26)]

    def blackhole(timeout):
        time.sleep(timeout)
        raise httpcore.ConnectTimeout("blackholed")

    backend = _Backend(blackhole)
    monkeypatch.setattr(pictures, "_network_backend", lambda: backend)
    assert _timed_refusal() < _BOUND
    assert all(timeout <= 0.5 for _, timeout in backend.dialed)


def test_at_most_eight_of_the_answers_are_dialled(net, monkeypatch):
    net.table[_AVATAR_HOST] = [f"93.184.215.{i}" for i in range(20, 32)]

    def refused(timeout):
        raise httpcore.ConnectError("refused")

    backend = _Backend(refused)
    monkeypatch.setattr(pictures, "_network_backend", lambda: backend)
    _timed_refusal()
    assert len(backend.dialed) == pictures.MAX_ADDRESSES < 12


def test_trickled_headers_are_cut_off_at_the_deadline(net, monkeypatch, short_deadline):
    head = (b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * 200
            + b"\r\nContent-Type: image/png\r\nContent-Length: 0\r\n\r\n")
    backend = _Backend(lambda timeout: _SlowStream(head))
    monkeypatch.setattr(pictures, "_network_backend", lambda: backend)
    assert _timed_refusal() < _BOUND


def test_a_stalled_tls_handshake_is_cut_off_at_the_deadline(net, monkeypatch, short_deadline):
    backend = _Backend(lambda timeout: _SlowStream(stall_tls=True))
    monkeypatch.setattr(pictures, "_network_backend", lambda: backend)
    assert _timed_refusal() < _BOUND


def test_a_stalled_request_write_is_cut_off_at_the_deadline(net, monkeypatch, short_deadline):
    backend = _Backend(lambda timeout: _SlowStream(b"HTTP/1.1 200 OK\r\n" * 50, stall_write=True))
    monkeypatch.setattr(pictures, "_network_backend", lambda: backend)
    assert _timed_refusal() < _BOUND


def test_a_hanging_lookup_is_abandoned_at_the_deadline(net, monkeypatch, short_deadline):
    release = threading.Event()
    lookup_threads: list[threading.Thread] = []

    def hang(host, port):
        lookup_threads.append(threading.current_thread())
        release.wait(10)
        raise OSError("released")

    monkeypatch.setattr(pictures, "_resolve", hang)
    try:
        assert _timed_refusal() < _BOUND
        # Abandoned on a daemon thread: a lookup that never returns cannot hold up interpreter exit.
        assert [t.daemon for t in lookup_threads] == [True]
    finally:
        release.set()


# ---- The same deadline over a real socket ----

class _ToListener(httpcore.NetworkBackend):
    """The real ``SyncBackend``, dialling the local listener whatever port the URL names (the fetch
    itself only ever asks for 443)."""

    def __init__(self, port: int) -> None:
        self._port, self._real = port, httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return self._real.connect_tcp(host, self._port, timeout=timeout,
                                      local_address=local_address, socket_options=socket_options)

    def sleep(self, seconds):  # pragma: no cover
        self._real.sleep(seconds)


@pytest.fixture
def listener(monkeypatch):
    """A TCP peer on 127.0.0.1 behaving as ``behaviour(conn)`` says, reachable only because this
    fixture lets loopback through ``_refusal``. It gives up after 4 s, so a fetch the deadline
    failed to bound still ends -- and fails the time bound."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    conns: list[socket.socket] = []
    state: Dict[str, Any] = {}

    def serve():
        conn, _ = srv.accept()
        conns.append(conn)
        try:
            state["behaviour"](conn)
        except OSError:
            pass

    monkeypatch.setattr(pictures, "_refusal", lambda address: None)
    monkeypatch.setattr(pictures, "_resolve", lambda host, port: ["127.0.0.1"])
    monkeypatch.setattr(pictures, "_network_backend", lambda: _ToListener(srv.getsockname()[1]))

    def start(behaviour):
        state["behaviour"] = behaviour
        threading.Thread(target=serve, daemon=True).start()

    yield start
    for conn in conns:
        conn.close()
    srv.close()


def _trickle_tls_record(conn: socket.socket) -> None:
    """Answer the ClientHello with a 16 KB handshake record header, then one byte every 50 ms --
    each byte well inside any per-read timeout."""
    conn.recv(4096)
    conn.sendall(b"\x16\x03\x03\x40\x00")
    end = time.monotonic() + 4
    while time.monotonic() < end:
        conn.sendall(b"\x00")
        time.sleep(0.05)


def _say_nothing(conn: socket.socket) -> None:
    conn.recv(4096)
    time.sleep(4)


@pytest.mark.parametrize("behaviour", [_trickle_tls_record, _say_nothing])
def test_a_real_peer_cannot_hold_the_fetch_past_the_deadline(listener, short_deadline, behaviour):
    listener(behaviour)
    assert _timed_refusal() < _BOUND


# ---------------------------------------------------------------------------
# The login round trip through the real provider and the gated app
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keys() -> Dict[str, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {
        "private_pem": key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode(),
        "public": key.public_key()}


def _provider(keys) -> oidc_plugin.SelfHostedOIDCProvider:
    p = oidc_plugin.SelfHostedOIDCProvider(issuer=_ISSUER, client_id=_CLIENT_ID)
    p._discovery = {
        "issuer": _ISSUER, "authorization_endpoint": f"{_ISSUER}/authorize",
        "token_endpoint": f"{_ISSUER}/token", "jwks_uri": f"{_ISSUER}/jwks",
        "revocation_endpoint": "", "token_endpoint_auth_methods_supported": []}
    p._discovery_fetched_at = time.time()
    signing_key = MagicMock()
    signing_key.key = keys["public"]
    jwks = MagicMock()
    jwks.get_signing_key_from_jwt.return_value = signing_key
    p._jwks_client = jwks
    return p


def _id_token(keys, *, sub: str, **claims: Any) -> str:
    now = int(time.time())
    body = {"iss": _ISSUER, "aud": _CLIENT_ID, "sub": sub, "iat": now, "exp": now + 900, **claims}
    return jwt.encode(body, keys["private_pem"], algorithm="RS256", headers={"kid": "k1"})


def _token_response(id_token: str) -> MagicMock:
    body = {"id_token": id_token, "token_type": "Bearer", "refresh_token": "rt"}
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = json.dumps(body)
    resp.json = MagicMock(return_value=body)
    resp.headers = {"content-type": "application/json"}
    return resp


@pytest.fixture
def app(keys):
    clear_providers()
    register_provider(_provider(keys))
    state = web_server.app.state
    prev = (getattr(state, "bound_host", None), getattr(state, "bound_port", None),
            getattr(state, "auth_required", None))
    state.bound_host, state.bound_port, state.auth_required = "gw.example.org", 443, True
    native_flow._reset_for_tests()
    yield lambda: TestClient(web_server.app, base_url="https://gw.example.org")
    native_flow._reset_for_tests()
    clear_providers()
    state.bound_host, state.bound_port, state.auth_required = prev


def _idp_state(start) -> str:
    assert start.status_code == 302, start.text
    return urllib.parse.parse_qs(urllib.parse.urlparse(start.headers["location"]).query)["state"][0]


def _callback(client: TestClient, keys, state: str, *, sub: str = "sam-sub",
              callback_extra: dict | None = None, **claims: Any):
    with patch("httpx.post", return_value=_token_response(_id_token(keys, sub=sub, **claims))):
        return client.get(
            "/auth/callback", params={"code": "c", "state": state, **(callback_extra or {})},
            follow_redirects=False)


def _login(client: TestClient, keys, **kwargs: Any):
    start = client.get("/auth/login", params={"provider": "self-hosted"}, follow_redirects=False)
    return _callback(client, keys, _idp_state(start), **kwargs)


_LOOPBACK = "http://127.0.0.1:53999/cb"
_VERIFIER = base64.urlsafe_b64encode(b"desktop-verifier-0123456789abcdef-0123456789").rstrip(b"=").decode()


def _native_start(browser: TestClient) -> str:
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(_VERIFIER.encode()).digest()).rstrip(b"=").decode()
    return _idp_state(browser.get("/auth/native/authorize", params={
        "provider": "self-hosted", "code_challenge": challenge, "code_challenge_method": "S256",
        "redirect_uri": _LOOPBACK, "state": "desk"}, follow_redirects=False))


SAM = {"name": "Sam Rivera", "email": "sam@example.org", "email_verified": True,
       "picture": _AVATAR_URL}


def test_email_and_picture_reach_auth_me_and_the_picture_is_served(app, keys, net):
    upstream = net.serve(_response(body=PNG, ctype="image/jpeg"))
    client = app()
    assert _login(client, keys, **SAM).status_code == 302

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "sam@example.org"
    assert body["display_name"] == "Sam Rivera"
    assert body["picture_url"] == "/api/auth/picture?id=self-hosted%3Asam-sub"
    # The provider's own URL is never handed to a client.
    assert _AVATAR_HOST not in me.text

    pic = client.get(body["picture_url"])
    assert pic.status_code == 200
    assert pic.content == PNG
    assert pic.headers["content-type"] == "image/png"  # from the bytes, not the upstream header
    assert pic.headers["x-content-type-options"] == "nosniff"
    assert pic.headers["content-security-policy"] == "default-src 'none'; sandbox"
    # Fetched once, at sign-in: serving it and verifying the session again never reach the host.
    client.get("/api/auth/me")
    assert len(upstream.dialed) == 1


def test_a_colleague_on_the_same_gateway_sees_the_picture_by_author_id(app, keys, net):
    net.serve(_response(body=PNG))
    _login(app(), keys, **SAM)

    colleague = app()
    _login(colleague, keys, sub="robin-sub", name="Robin")
    pic = colleague.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"})
    assert pic.status_code == 200 and pic.content == PNG
    # A colleague gets the picture by id and nothing else: auth/me is only ever about yourself.
    assert "sam@example.org" not in colleague.get("/api/auth/me").text


@pytest.mark.parametrize("flag", [False, "false"])
def test_an_unverified_email_is_not_in_auth_me(app, keys, net, flag):
    client = app()
    _login(client, keys, email="sam@example.org", email_verified=flag)
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == ""
    assert "sam@example.org" not in me.text


def test_no_picture_claim_means_no_picture_and_no_error(app, keys, net):
    upstream = net.serve(_response(body=PNG))
    client = app()
    assert _login(client, keys, name="Sam Rivera").status_code == 302
    body = client.get("/api/auth/me").json()
    assert "picture_url" not in body
    assert upstream.dialed == []
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).status_code == 404


@pytest.mark.parametrize("serve", [
    lambda net: net.serve(),                                         # upstream unreachable
    lambda net: net.serve(_response(500, body=b"boom")),
    lambda net: net.serve(_response(body=b"<html>", ctype="text/html")),
    lambda net: net.table.__setitem__(_AVATAR_HOST, ["10.0.0.5"]),   # resolves privately
    lambda net: net.table.pop(_AVATAR_HOST),                         # does not resolve
])
def test_a_failed_fetch_still_lets_login_succeed(app, keys, net, serve):
    serve(net)
    client = app()
    resp = _login(client, keys, **SAM)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "sam@example.org"
    assert "picture_url" not in me.json()


def test_the_picture_endpoint_refuses_an_unauthenticated_request(app, keys, net):
    net.serve(_response(body=PNG))
    _login(app(), keys, **SAM)
    stranger = app()
    resp = stranger.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"})
    assert resp.status_code == 401
    assert resp.content != PNG


def test_every_id_without_a_stored_picture_gets_the_same_404(app, keys, net):
    net.serve(_response(body=PNG))
    client = app()
    _login(client, keys, **SAM)
    answers = [
        client.get("/api/auth/picture"),
        client.get("/api/auth/picture", params={"id": ""}),
        client.get("/api/auth/picture", params={"id": "self-hosted:nobody"}),
        client.get("/api/auth/picture", params={"id": "../../config.yaml"}),
        client.get("/api/auth/picture", params={"id": "stub:sam-sub"}),
    ]
    assert {(r.status_code, r.content) for r in answers} == {(answers[0].status_code, answers[0].content)}
    assert answers[0].status_code == 404


def test_client_supplied_email_or_picture_is_ignored(app, keys, net):
    upstream = net.serve(_response(body=PNG))
    client = app()
    # The IdP asserts no picture and no email; the client adds both on the callback and on auth/me.
    resp = _login(client, keys, name="Sam Rivera", callback_extra={
        "picture": _AVATAR_URL, "email": "eve@example.org", "email_verified": "true"})
    assert resp.status_code == 302
    assert upstream.dialed == []
    me = client.get("/api/auth/me", params={"email": "eve@example.org", "picture_url": _AVATAR_URL})
    assert me.json()["email"] == ""
    assert "picture_url" not in me.json()
    assert "eve@example.org" not in me.text
    # And there is no way to hand the gateway a picture at all.
    assert client.post("/api/auth/picture", params={"id": "self-hosted:sam-sub"},
                       content=PNG, headers={"content-type": "image/png"}).status_code == 405
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).status_code == 404


def test_a_second_login_replaces_the_stored_picture(app, keys, net):
    client = app()
    net.serve(_response(body=PNG))
    _login(client, keys, **SAM)
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).content == PNG

    net.serve(_response(body=PNG_B))
    _login(client, keys, **SAM)
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).content == PNG_B

    # A login that brings no picture leaves none behind: only what this login's provider sent.
    _login(client, keys, **{k: v for k, v in SAM.items() if k != "picture"})
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).status_code == 404
    assert "picture_url" not in client.get("/api/auth/me").json()

    # Nor does a login whose fetch fails.
    net.serve(_response(body=PNG))
    _login(client, keys, **SAM)
    net.serve(_response(500, body=b"boom"))
    _login(client, keys, **SAM)
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).status_code == 404


def test_the_picture_is_stored_under_the_gateway_home_by_hashed_identity(app, keys, net):
    net.serve(_response(body=PNG))
    _login(app(), keys, **SAM)
    from hermes_constants import get_hermes_home
    stored = list((get_hermes_home() / "dashboard_auth" / "pictures").iterdir())
    assert len(stored) == 1
    assert stored[0].name == hashlib.sha256(b"self-hosted:sam-sub").hexdigest()
    assert stored[0].read_bytes() == PNG


def test_the_login_returns_on_time_while_the_fetch_thread_is_still_stuck(app, keys, net, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    limiters: list = []
    real_run_sync = anyio.to_thread.run_sync

    async def spy(func, *args, limiter=None, **kwargs):
        if getattr(func, "__self__", None).__class__.__name__ == "_LoginFetch":
            limiters.append(limiter)
        return await real_run_sync(func, *args, limiter=limiter, **kwargs)

    def stuck(url, deadline=None):
        entered.set()
        release.wait(10)
        finished.set()
        raise pictures.PictureRefused("released")

    monkeypatch.setattr(pictures, "fetch_picture", stuck)
    monkeypatch.setattr(pictures, "LOGIN_WAIT_SEC", 0.3)
    monkeypatch.setattr(anyio.to_thread, "run_sync", spy)
    client = app()
    try:
        started = time.monotonic()
        resp = _login(client, keys, **SAM)
        assert resp.status_code == 302
        assert time.monotonic() - started < _BOUND
        assert entered.is_set() and not finished.is_set()
        # The stuck fetch runs under the pictures' own limiter, never on the dashboard's shared
        # threadpool tokens, and holds exactly one slot for this person.
        assert limiters == [pictures._LOGIN_LIMITER]
        assert pictures._in_flight == {"self-hosted:sam-sub"}
        me = client.get("/api/auth/me")
        assert me.status_code == 200 and "picture_url" not in me.json()
    finally:
        release.set()


def test_a_native_login_stores_the_picture_and_the_desktop_bearer_sees_it(app, keys, net):
    net.serve(_response(body=PNG))
    browser = app()
    resp = _callback(browser, keys, _native_start(browser), **SAM)
    assert resp.status_code == 302
    loopback = urllib.parse.urlparse(resp.headers["location"])
    assert f"{loopback.scheme}://{loopback.netloc}{loopback.path}" == _LOOPBACK
    code = urllib.parse.parse_qs(loopback.query)["code"][0]

    desktop = app()
    tokens = desktop.post("/auth/native/token", json={"code": code, "code_verifier": _VERIFIER})
    assert tokens.status_code == 200, tokens.text
    bearer = {"Authorization": f"Bearer {tokens.json()['access_token']}"}
    me = desktop.get("/api/auth/me", headers=bearer).json()
    assert me["email"] == "sam@example.org"
    assert desktop.get(me["picture_url"], headers=bearer).content == PNG


def test_a_failed_native_login_leaves_the_stored_picture_alone(app, keys, net):
    client = app()
    net.serve(_response(body=PNG))
    _login(client, keys, **SAM)

    upstream = net.serve(_response(body=PNG_B))
    browser = app()
    state = _native_start(browser)
    native_flow._reset_for_tests()  # the desktop's pending authorization is gone
    assert _callback(browser, keys, state, **SAM).status_code == 400
    assert upstream.dialed == []
    assert client.get("/api/auth/picture", params={"id": "self-hosted:sam-sub"}).content == PNG


# ---------------------------------------------------------------------------
# Login-time fetches: never a wait for a worker, one per person, the last sign-in wins
# ---------------------------------------------------------------------------


class _FakeFetch:
    """``fetch_picture`` stand-in: an ``/slow`` URL blocks until released, a URL ending in ``/fails``
    fails, anything else returns a picture unique to its URL."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[str] = []
        self.deadlines: list = []

    def __call__(self, url, deadline=None):
        self.calls.append(url)
        self.deadlines.append(deadline)
        if "/slow" in url:
            self.release.wait(10)
        if url.endswith("/fails"):
            raise pictures.PictureRefused("failed")
        return png(pad=url.encode()), "image/png"


@pytest.fixture
def fake_fetch(monkeypatch):
    fake = _FakeFetch()
    monkeypatch.setattr(pictures, "fetch_picture", fake)
    # Long enough that a login held by a busy pool would blow the time bound.
    monkeypatch.setattr(pictures, "LOGIN_WAIT_SEC", 5.0)
    monkeypatch.setattr(pictures, "_in_flight", set())
    monkeypatch.setattr(pictures, "_latest_generation", {})
    yield fake
    fake.release.set()
    deadline = time.monotonic() + 5
    while pictures._in_flight and time.monotonic() < deadline:
        time.sleep(0.01)


def _sign_in(user: str, url: str = ""):
    return SimpleNamespace(provider="oidc", user_id=user, picture=url)


def _stored(user: str):
    found = pictures.read_picture(f"oidc:{user}")
    return found[0] if found else None


async def _until(predicate) -> None:
    with anyio.fail_after(3):
        while not predicate():
            await anyio.sleep(0.01)


def test_a_colleague_signs_in_promptly_while_slow_fetches_hold_every_slot(fake_fetch):
    async def main():
        await pictures.store_login_picture(_sign_in("robin", "https://fast/robin-1"))
        async with anyio.create_task_group() as tg:
            for i in range(pictures.LOGIN_SLOTS):
                tg.start_soon(pictures.store_login_picture, _sign_in(f"eve{i}", "https://slow/eve"))
            await _until(lambda: fake_fetch.calls.count("https://slow/eve") == pictures.LOGIN_SLOTS)
            started = time.monotonic()
            await pictures.store_login_picture(_sign_in("robin", "https://fast/robin-2"))
            waited = time.monotonic() - started
            fake_fetch.release.set()
        return waited

    assert anyio.run(main) < _BOUND
    assert "https://fast/robin-2" not in fake_fetch.calls
    # A busy gateway skips the refresh; it never deletes or replaces the picture already kept.
    assert _stored("robin") == png(pad=b"https://fast/robin-1")


def test_a_second_sign_in_by_the_same_person_does_not_start_a_second_fetch(fake_fetch):
    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(pictures.store_login_picture, _sign_in("sam", "https://slow/sam"))
            await _until(lambda: fake_fetch.calls == ["https://slow/sam"])
            started = time.monotonic()
            await pictures.store_login_picture(_sign_in("sam", "https://fast/sam"))
            waited = time.monotonic() - started
            fake_fetch.release.set()
        return waited

    assert anyio.run(main) < _BOUND
    assert fake_fetch.calls == ["https://slow/sam"]


@pytest.mark.parametrize("stalled", ["https://slow/fails", "https://slow/other-face"])
def test_an_older_stalled_fetch_cannot_overwrite_or_delete_what_a_newer_sign_in_kept(
        fake_fetch, monkeypatch, stalled):
    monkeypatch.setattr(pictures, "LOGIN_WAIT_SEC", 0.2)

    async def main():
        await pictures.store_login_picture(_sign_in("sam", "https://fast/sam-0"))
        await pictures.store_login_picture(_sign_in("sam", stalled))  # the login gives up; fetch stalls
        await pictures.store_login_picture(_sign_in("sam", "https://fast/sam-2"))  # newer sign-in
        fake_fetch.release.set()
        await _until(lambda: not pictures._in_flight)

    anyio.run(main)
    assert _stored("sam") == png(pad=b"https://fast/sam-0")


def test_a_fetch_finishing_after_its_login_gave_up_stores_only_while_still_the_latest(
        fake_fetch, monkeypatch):
    monkeypatch.setattr(pictures, "LOGIN_WAIT_SEC", 0.2)

    async def main():
        # Still the latest sign-in when it finishes: the late result is kept.
        await pictures.store_login_picture(_sign_in("sam", "https://slow/late-1"))
        fake_fetch.release.set()
        await _until(lambda: not pictures._in_flight)
        kept = _stored("sam")
        # A newer sign-in happened meanwhile: the late result is dropped.
        fake_fetch.release.clear()
        await pictures.store_login_picture(_sign_in("sam", "https://slow/late-2"))
        await pictures.store_login_picture(_sign_in("sam", "https://fast/sam-3"))
        fake_fetch.release.set()
        await _until(lambda: not pictures._in_flight)
        return kept

    assert anyio.run(main) == png(pad=b"https://slow/late-1")
    assert _stored("sam") == png(pad=b"https://slow/late-1")


def test_the_fetch_deadline_counts_from_the_sign_in(fake_fetch):
    async def main():
        handed_over = time.monotonic()
        await pictures.store_login_picture(_sign_in("sam", "https://fast/sam"))
        return handed_over

    handed_over = anyio.run(main)
    (deadline,) = fake_fetch.deadlines
    assert handed_over + pictures._DEADLINE_SEC <= deadline < handed_over + pictures._DEADLINE_SEC + 1
