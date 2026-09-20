"""Explicit shared-store links preserve OAuth rotation and one kernel lock."""
import json
import threading
from contextlib import contextmanager

from hermes_constants import set_hermes_home_override, reset_hermes_home_override


@contextmanager
def home_scope(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def make_homes(tmp_path):
    owner = tmp_path / 'owner'
    owner.mkdir()
    (owner / 'auth.json').write_text(json.dumps({'version': 1, 'providers': {
        'openai-codex': {'tokens': {'access_token': 'old-access', 'refresh_token': 'old-refresh'}}}}))
    homes = []
    for name in ('a', 'b'):
        home = tmp_path / name
        home.mkdir()
        (home / 'auth.json').symlink_to(owner / 'auth.json')
        homes.append(home)
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    return owner, homes, isolated


def test_rotation_adopted_across_profiles_without_copy_or_replay(tmp_path, monkeypatch):
    from hermes_cli import auth
    from hermes_cli.auth_codex import _refresh_codex_auth_tokens
    owner, (a, b), isolated = make_homes(tmp_path)
    calls = []
    def rotate(access, refresh, **kwargs):
        calls.append(refresh)
        return {'access_token': 'new-access', 'refresh_token': 'new-refresh'}
    monkeypatch.setattr(auth, 'refresh_codex_oauth_pure', rotate)
    old = {'access_token': 'old-access', 'refresh_token': 'old-refresh'}
    for home in (a, b, a):
        with home_scope(home):
            result = _refresh_codex_auth_tokens(old, 1)
            assert result['refresh_token'] == 'new-refresh'
            assert auth._auth_file_path() == owner / 'auth.json'
        assert (home / 'auth.json').is_symlink()
    assert calls == ['old-refresh']
    with home_scope(isolated):
        assert auth._load_auth_store()['providers'] == {}


def test_shared_profiles_contend_for_same_lock(tmp_path):
    from hermes_cli import auth
    owner, (a, b), isolated = make_homes(tmp_path)
    attempted = threading.Event()
    timed_out = threading.Event()
    errors = []
    def contender():
        try:
            with home_scope(b):
                attempted.set()
                try:
                    with auth._auth_store_lock(timeout_seconds=1):
                        errors.append('second profile entered held lock')
                except TimeoutError:
                    timed_out.set()
        except Exception as exc:
            errors.append(str(exc))
    with home_scope(a), auth._auth_store_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        assert attempted.wait(5)
        thread.join(5)
        assert not thread.is_alive()
        assert timed_out.is_set()
        assert errors == []
    with home_scope(b), auth._auth_store_lock(timeout_seconds=1):
        assert auth._auth_file_path() == owner / 'auth.json'
