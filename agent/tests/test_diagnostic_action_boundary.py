import pytest
import gen2b_agent as g

@pytest.mark.asyncio
async def test_diagnostic_tools_cannot_execute_side_effects():
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = 'diagnostic-pipeline-unit-test'
    agent._owner_channel = 'web'
    # No context or initialized business state: any access past the guard fails.
    assert 'диагностическом' in await agent.ask_owner(None, 'вопрос')
    assert 'диагностическом' in await agent.finalize_call(None, 'agreed', 'итог')
    assert 'диагностическом' in await agent.end_call(None)

def test_business_rooms_are_not_restricted():
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = 'business-call'
    agent._owner_channel = 'web'
    assert not agent._diagnostic_tool_blocked()
    agent._room_name = 'diagnostic-pipeline-test'
    agent._owner_channel = 'telegram'
    assert not agent._diagnostic_tool_blocked()
