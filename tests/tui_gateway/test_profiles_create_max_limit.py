"""``profiles.max`` enforced through the ``profiles.create`` RPC — the desktop/ws twin of the
CLI's ``hermes profile create`` and the dashboard's ``POST /api/profiles``. All three call the
same chokepoint, ``hermes_cli.profiles.create_profile()``; this test pins that the RPC surfaces
its refusal the same way the other two do (a JSON-RPC error, not a raised exception or a silent
200), with the same wording.
"""

from hermes_constants import get_hermes_home


def _create(params: dict) -> dict:
    from tui_gateway import server
    handler = server._methods["profiles.create"]
    return handler("r1", params)


def test_refused_at_the_limit_returns_a_json_rpc_error_naming_the_limit_and_the_count():
    (get_hermes_home() / "config.yaml").write_text("profiles:\n  max: 1\n", encoding="utf-8")

    resp = _create({"name": "onemore", "no_alias": True})

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "r1"
    assert "result" not in resp
    error = resp["error"]
    assert error["code"] == 4062
    assert error["message"] == (
        "This gateway allows 1 profiles and already has 1. "
        "Delete a profile first, or raise profiles.max in config.yaml."
    )
    assert not (get_hermes_home() / "profiles" / "onemore").exists()


def test_succeeds_below_the_limit():
    (get_hermes_home() / "config.yaml").write_text("profiles:\n  max: 2\n", encoding="utf-8")

    resp = _create({"name": "roomleft", "no_alias": True})

    assert resp.get("error") is None
    assert resp["result"]["ok"] is True
    assert (get_hermes_home() / "profiles" / "roomleft").is_dir()


def test_unset_limit_is_unlimited():
    resp = _create({"name": "anyname", "no_alias": True})

    assert resp.get("error") is None
    assert resp["result"]["ok"] is True
