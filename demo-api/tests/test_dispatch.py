import sys
from pathlib import Path

import pytest

from dispatch import CallWithPersonaDispatcher
from models import CallRequest


def request():
    return CallRequest(
        contact_name="Cafe", phone_number="+77001234567", task="Book a table",
        details="Window seat", demo_session_id="demo_1",
    )


@pytest.mark.asyncio
async def test_dispatch_adapter_imports_and_calls_async_dispatch_function(tmp_path, monkeypatch):
    module = tmp_path / "call_with_persona.py"
    module.write_text(
        "async def dispatch_call(**kwargs):\n"
        "    open(kwargs.pop('marker'), 'w').write(repr(kwargs))\n"
        "    return {'room_name': kwargs['room_name'], 'sip_call_id': 'sip-1'}\n"
    )
    marker = tmp_path / "called"
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("call_with_persona", None)
    dispatcher = CallWithPersonaDispatcher(extra_kwargs={"marker": str(marker)})

    result = await dispatcher.dispatch(request(), session_id="demo_1", room_name="room-1")

    assert result.room_name == "room-1"
    assert result.call_id == "sip-1"
    called = marker.read_text()
    assert "armen_personal_assistant" in called
    assert "web_demo" in called


def test_dispatch_adapter_reports_missing_reusable_function(tmp_path, monkeypatch):
    (tmp_path / "call_with_persona.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("call_with_persona", None)
    with pytest.raises(RuntimeError, match="reusable async dispatch"):
        CallWithPersonaDispatcher()


@pytest.mark.asyncio
async def test_hangup_adapter_removes_current_sip_participant(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    control = AsyncMock()
    monkeypatch.setitem(sys.modules, 'call_control', SimpleNamespace(hang_up_sip_participant=control))
    dispatcher = object.__new__(CallWithPersonaDispatcher)
    await dispatcher.hangup(room_name='actual-room')
    control.assert_awaited_once_with('actual-room', 'callee')
