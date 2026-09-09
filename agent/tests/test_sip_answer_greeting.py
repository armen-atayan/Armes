import asyncio
from types import SimpleNamespace

import pytest

import gen2b_agent


class FakeRoom:
    def __init__(self, status):
        self.remote_participants = {
            "callee": SimpleNamespace(attributes={"sip.callStatus": status})
        }


@pytest.mark.asyncio
async def test_waits_for_active_sip_call_before_greeting():
    room = FakeRoom("dialing")
    calls = []

    async def waiter(room, *, identity, attribute, value):
        calls.append((identity, attribute, value))

    await gen2b_agent.wait_for_sip_call_active(room, "callee", waiter=waiter)

    assert calls == [("callee", "sip.callStatus", "active")]


@pytest.mark.asyncio
async def test_does_not_wait_when_sip_call_is_already_active():
    room = FakeRoom("active")

    async def waiter(*args, **kwargs):
        raise AssertionError("must not wait")

    await gen2b_agent.wait_for_sip_call_active(room, "callee", waiter=waiter)
