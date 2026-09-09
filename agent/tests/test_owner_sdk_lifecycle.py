"""Regression using installed LiveKit executor, not a direct _func call."""
import asyncio
from unittest.mock import Mock
import pytest
from livekit.agents import AgentSession, RunContext
from livekit.agents.llm import FunctionCall
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.tool_executor import _ToolExecutor
import gen2b_agent as g
import live_callback

@pytest.mark.asyncio
async def test_sdk_drain_cancels_owner_wait_and_rejects_late_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(g,'LIVE_CALLBACK_DIR',tmp_path)
    monkeypatch.setattr(g,'LIVE_CALLBACK_TIMEOUT',60)
    monkeypatch.setattr(g,'TELEGRAM_CHAT_ID','135001671')
    sent=asyncio.Event()
    ids=[]
    loop=asyncio.get_running_loop()
    def send(rid, question):
        ids.append(rid)
        loop.call_soon_threadsafe(sent.set)
        return 123
    monkeypatch.setattr(g,'send_live_callback_to_telegram',send)
    agent=g.Gen2BAssistant(room_name='sdk-drain',participant_identity='test',
        on_outcome_saved=lambda:None, system_prompt='test',recording_path_getter=lambda:None)
    session=AgentSession()
    monkeypatch.setattr(session,'say',Mock())
    ctx=RunContext(session=session,speech_handle=SpeechHandle.create(),
        function_call=FunctionCall(call_id='owner-call',name='ask_owner',arguments='{}'))
    executor=_ToolExecutor()
    dispatch=asyncio.create_task(executor.execute(tool=agent.ask_owner,run_ctx=ctx,
        raw_arguments={'question':'Изменить адрес?'}))
    try:
        await asyncio.wait_for(sent.wait(),1)
        drain=asyncio.create_task(executor.drain())
        done,_=await asyncio.wait([drain],timeout=.3)
        assert drain in done, 'LiveKit drain waits for owner timeout instead of cancelling the tool'
        assert not list((tmp_path/'pending').glob('*.json'))
        assert not live_callback.resolve_request(tmp_path,ids[0],'yes',chat_id='135001671')
        assert await dispatch is None
    finally:
        await executor.aclose()
        await asyncio.gather(dispatch,return_exceptions=True)
        if 'drain' in locals():await asyncio.gather(drain,return_exceptions=True)
