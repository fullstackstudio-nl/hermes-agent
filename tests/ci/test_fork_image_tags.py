"""The fork image workflow's tag choice: `:main` only for builds of main, never for a manual
dispatch of another ref (which would race, or roll back, what a push to main published)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "fork-image.yml"
IMAGE = "ghcr.io/fullstackstudio-org/hermes-agent"


def _step(step_id: str) -> dict:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    return next(s for s in steps if s.get("id") == step_id)


def _outputs(tmp_path, event: str, input_ref: str) -> dict:
    out = tmp_path / "out"
    out.write_text("", encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(out), "IMAGE": IMAGE, "EVENT": event,
           "INPUT_REF": input_ref, "SHORT": "abc1234", "VERSION": "0.21.5", "PLUGIN_TAG": "v0.9.0"}
    subprocess.run(["bash", "-e", "-c", _step("tags")["run"]], env=env, check=True)
    text = out.read_text(encoding="utf-8")
    head, _, rest = text.partition("list<<EOF\n")
    tags, _, tail = rest.partition("EOF\n")
    return {"tags": [t for t in tags.splitlines() if t], "main": tail.strip()}


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.mark.parametrize("event, input_ref", [("push", ""), ("schedule", ""), ("workflow_dispatch", "")])
def test_builds_of_main_move_main(tmp_path, event, input_ref):
    result = _outputs(tmp_path, event, input_ref)
    assert f"{IMAGE}:main" in result["tags"] and result["main"] == "main=true"
    assert f"{IMAGE}:abc1234-plugin-v0.9.0" in result["tags"]


def test_dispatch_of_another_ref_never_moves_main(tmp_path):
    result = _outputs(tmp_path, "workflow_dispatch", "0123456789abcdef0123456789abcdef01234567")
    assert f"{IMAGE}:main" not in result["tags"] and result["main"] == "main=false"
    assert result["tags"] == [f"{IMAGE}:abc1234", f"{IMAGE}:0.21.5-abc1234", f"{IMAGE}:abc1234-plugin-v0.9.0"]


def test_build_step_takes_the_computed_tags():
    build = next(s for s in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
                 if s.get("id") == "build")
    assert build["with"]["tags"] == "${{ steps.tags.outputs.list }}"
