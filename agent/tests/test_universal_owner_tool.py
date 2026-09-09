import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import gen2b_agent as g
import live_callback


def test_every_non_exact_persona_receives_owner_tool_policy():
    personas = g.load_personas()
    for persona_key, persona in personas.items():
        config = g.resolve_call_config(json.dumps({'persona': persona_key}))
        if persona.get('exact_system_prompt', False):
            assert 'УНИВЕРСАЛЬНОЕ УТОЧНЕНИЕ' not in config['system_prompt']
        else:
            assert 'УНИВЕРСАЛЬНОЕ УТОЧНЕНИЕ' in config['system_prompt']
            assert 'ask_owner' in config['system_prompt']


def test_owner_policy_forbids_choosing_unspecified_alternatives():
    config = g.resolve_call_config(json.dumps({
        'persona': 'armen_personal_assistant',
        'task': 'Забронировать стол на завтра в 20:00',
    }))
    prompt = config['system_prompt']
    assert 'не выбирай ни один вариант самостоятельно' in prompt.lower()
    assert 'options' in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('answer,action', [('yes','approve'), ('no','decline'), ('Проверь доставку на завтра','instruct'), ('timeout','timeout')])
async def test_generic_tool_scopes_decision_and_preserves_context(monkeypatch, tmp_path, answer, action):
    monkeypatch.setattr(g, 'LIVE_CALLBACK_DIR', tmp_path)
    monkeypatch.setattr(g, 'TELEGRAM_CHAT_ID', '135001671')
    monkeypatch.setattr(g, 'LIVE_CALLBACK_TIMEOUT', .05)
    def send(rid, question):
        assert 'Доставка' in question
        assert 'Курьер предлагает' in question
        if answer != 'timeout':
            live_callback.resolve_request(tmp_path, rid, answer)
        return 123
    monkeypatch.setattr(g, 'send_live_callback_to_telegram', send)
    async def never_finishes():
        await asyncio.Event().wait()
    speech = SimpleNamespace(wait_for_playout=never_finishes)
    session = SimpleNamespace(
        say=Mock(return_value=speech), generate_reply=Mock(), interrupt=AsyncMock()
    )
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = 'generic-delivery-test'
    result = await asyncio.wait_for(g.Gen2BAssistant.ask_owner._func(
        agent, SimpleNamespace(session=session),
        question='Доставка завтра в 18:00 подходит?',
        context='Курьер предлагает другой день вместо сегодня.',
    ), timeout=1)
    result = json.loads(result)
    await asyncio.sleep(0)
    assert result['action'] == action
    assert result['question'] == 'Доставка завтра в 18:00 подходит?'
    assert result['response'] == answer
    session.generate_reply.assert_called_once()
    continuation = session.generate_reply.call_args.kwargs['user_input']
    assert f"Ответ Армена: {answer}" in continuation
    assert session.generate_reply.call_args.kwargs['instructions'].startswith("Немедленно продолжи")
    assert session.generate_reply.call_args.kwargs['allow_interruptions'] is False
    session.interrupt.assert_awaited_once_with(force=True)
    assert 'только' in result['instruction'] if action == 'approve' else True
    assert not list((tmp_path/'pending').glob('*.json'))


@pytest.mark.asyncio
async def test_send_failure_is_not_approval_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(g,'LIVE_CALLBACK_DIR',tmp_path)
    monkeypatch.setattr(g,'TELEGRAM_CHAT_ID','135001671')
    def fail(*args):
        raise RuntimeError('network unavailable')
    monkeypatch.setattr(g,'send_live_callback_to_telegram',fail)
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name='failed-test'
    ctx=SimpleNamespace(session=SimpleNamespace(say=Mock()))
    result=json.loads(await g.Gen2BAssistant.ask_owner._func(agent,ctx,'Изменить адрес?'))
    assert result['action']=='delivery_failed'
    assert not list((tmp_path/'pending').glob('*.json'))


@pytest.mark.asyncio
async def test_owner_continuation_uses_supplied_name_without_inventing_full_name_requirement():
    session = SimpleNamespace(interrupt=AsyncMock(), generate_reply=Mock())
    decision = json.loads(g.decision_result(
        'Разрешаете оформить бронь на имя Армен?', '', 'Оформить на имя Армен'))
    await g.resume_owner_turn(session, decision)
    instructions = session.generate_reply.call_args.kwargs['instructions']
    assert 'Не переспрашивай уже полученные сведения' in instructions
    assert 'полное имя' in instructions
    assert 'передай' in instructions.lower()
    assert 'не подтверждение собеседника' in instructions.lower()
