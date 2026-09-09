"""Production deposit regression: rendered contract and offline tool dispatch.

The expected tool call is synthetic. These tests do not measure LLM compliance
or contact a model provider, Telegram, or the demo API.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from livekit.agents import AgentSession, RunContext
from livekit.agents.llm import FunctionCall, utils
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.tool_executor import _ToolExecutor

import gen2b_agent as g
import live_callback


@pytest.fixture
def deposit_probe():
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                         'owner_deposit_clarification.json').read_text())
    config = g.resolve_call_config(json.dumps(fixture['metadata']))
    fixture['request'] = {
        'model': config['llm_model'],
        'messages': [{'role': 'system', 'content': config['system_prompt']},
                     *fixture['messages']],
        'tools': [utils.build_legacy_openai_schema(object.__new__(g.Gen2BAssistant).ask_owner)],
        'tool_choice': 'auto',
    }
    return fixture


def test_rendered_deposit_prompt_requires_tool_without_holding_prose(deposit_probe):
    prompt = deposit_probe['request']['messages'][0]['content']
    assert 'немедленно вызови ask_owner в этом же ходе без предварительной реплики' in prompt
    assert '«размер депозита мне нужно уточнить»' in prompt
    assert 'Не спрашивай у собеседника сумму, которую должен выбрать или разрешить Армен' in prompt
    assert 'Минимум «от 10 000» не означает разрешение Армена на точную сумму' in prompt
    assert 'Фразу ожидания произноси только когда' not in prompt
    assert 'Задавай только один вопрос в каждой реплике и жди ответа' in prompt
    assert 'Не спрашивай о ценах, стоимости, депозитах, предоплате или минимальном чеке' in prompt
    assert 'Какую точную сумму и валюту вы называете для депозита?' in prompt


def test_provider_visible_owner_tool_contract_requires_immediate_silent_call(deposit_probe):
    tool = deposit_probe['request']['tools'][0]['function']
    assert tool['name'] == 'ask_owner'
    assert tool['parameters']['required'] == ['question']
    description = tool['description']
    assert 'немедленно вызови ask_owner в этом же ходе без предварительной реплики' in description
    assert 'точная сумма депозита' in description
    assert 'не спрашивай эти сведения у собеседника повторно' in description.lower()
    assert 'Только инструмент произносит «Секундочку, сейчас уточню»' in description


@pytest.mark.asyncio
async def test_offline_deposit_tool_call_publishes_question_and_speaks_once(
    deposit_probe, monkeypatch, tmp_path,
):
    monkeypatch.setattr(g, 'LIVE_CALLBACK_DIR', tmp_path)
    monkeypatch.setattr(g, 'LIVE_CALLBACK_TIMEOUT', .2)
    monkeypatch.setattr(g, 'resume_owner_turn', AsyncMock())
    telegram = Mock(side_effect=AssertionError('Offline web probe must not send Telegram'))
    monkeypatch.setattr(g, 'send_live_callback_to_telegram', telegram)
    events = []

    def emit(sid, room, kind, payload):
        events.append((kind, payload))
        if kind == 'owner.question':
            live_callback.resolve_request(tmp_path, payload['request_id'], '10 тысяч')

    monkeypatch.setattr(g, 'emit_demo_event', emit)
    agent = g.Gen2BAssistant(
        room_name='offline-deposit', participant_identity='test',
        on_outcome_saved=lambda: None, recording_path_getter=lambda: None,
        system_prompt=deposit_probe['request']['messages'][0]['content'],
    )
    agent._owner_channel = 'web'
    agent._demo_session_id = 'offline-deposit'
    session = AgentSession()
    monkeypatch.setattr(session, 'say', Mock())
    call = deposit_probe['expected_tool_call']
    args = json.loads(call['function']['arguments'])
    ctx = RunContext(
        session=session, speech_handle=SpeechHandle.create(),
        function_call=FunctionCall(call_id=call['id'], **call['function']),
    )
    executor = _ToolExecutor()
    try:
        await executor.execute(tool=agent.ask_owner, run_ctx=ctx, raw_arguments=args)
        await asyncio.sleep(0)
    finally:
        await executor.aclose()
    assert [kind for kind, _ in events] == ['owner.question', 'owner.answer']
    assert events[0][1]['question'] == args['question']
    assert args['question'].count('?') == 1
    assert events[1][1]['answer'] == '10 тысяч'
    session.say.assert_called_once_with('Секундочку, сейчас уточню.', allow_interruptions=False)
    telegram.assert_not_called()
