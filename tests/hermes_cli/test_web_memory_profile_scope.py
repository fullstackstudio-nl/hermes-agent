"""Memory status and activation bind the requested profile even after multiplexing starts."""
import asyncio
import json
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_cli.web_models import MemoryProviderSelect
from hermes_cli.web_routers import ops
from tui_gateway import launch_profile_policy


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    other = home / 'profiles' / 'other'
    other.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(launch_profile_policy, '_snapshot', None)
    for directory, mode in ((home, 'oss'), (other, 'platform')):
        (directory / 'config.yaml').write_text('memory:\n  provider: mem0\n')
        (directory / '.env').write_text('')
        (directory / 'mem0.json').write_text(json.dumps({
            'mode': mode, 'oss': {'vector_store': {'provider': 'qdrant'}}
        }))
    secret_scope.set_multiplex_active(True)
    yield home, other
    secret_scope.set_multiplex_active(False)


def test_status_preserves_oss_availability_across_profile_switches(homes):
    async def run():
        for profile, expected in ((None, True), ('other', False), (None, True)):
            response = await ops.get_memory_status() if profile is None else await ops.get_memory_status(profile)
            mem0 = next(row for row in response['providers'] if row['name'] == 'mem0')
            assert mem0['available'] is expected
            assert mem0['configured'] is expected
    asyncio.run(run())


def test_activation_resolves_provider_under_own_profile(homes):
    async def run():
        body = MemoryProviderSelect(provider='mem0')
        assert (await ops.set_memory_provider(body))['active'] == 'mem0'
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as error:
            await ops.set_memory_provider(body, 'other')
        assert error.value.status_code == 400
        assert (await ops.set_memory_provider(body))['active'] == 'mem0'
    asyncio.run(run())
