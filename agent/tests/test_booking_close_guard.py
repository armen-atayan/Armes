from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
import gen2b_agent as g


def test_booking_guard_is_scoped_to_personal_assistant_booking_tasks():
    import json
    booked=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','task':'Забронируй стол на четверых.'}))
    appointment=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','task':'Позвони в барбершоп и запиши меня на стрижку завтра в 15:00.'}))
    other=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','task':'Узнай адрес офиса.'}))
    assert booked.get('booking_confirmation_required') is True
    assert appointment.get('booking_confirmation_required') is True
    assert other.get('booking_confirmation_required') is False


def agent_with_text(text, enabled=True):
    a=g.Gen2BAssistant(room_name='guard-test', participant_identity='test',
        on_outcome_saved=lambda:None, system_prompt='Стол на четверых.',recording_path_getter=lambda:None)
    a._booking_confirmation_required=enabled
    a._chat_ctx.add_message(role='user',content=text)
    return a


@pytest.mark.asyncio
async def test_production_booking_confirmation_survives_three_final_fragments():
    # demo_7c9319d611abb17b: these are finals within ONE callee turn.
    from livekit.agents.llm import ChatMessage
    a = agent_with_text('')
    a._chat_ctx.add_message(role='assistant', content='Запишите, пожалуйста, на имя Армен')
    fragments = [
        'Хорошо записываю на имя армен.',
        'На завтра к ирине на двадцать ноль ноль.',
        'Спасибо.',
    ]
    for text in fragments:
        a._record_callee_final(text)
    await a.on_user_turn_completed(
        a.chat_ctx.copy(), ChatMessage(role='user', content=[' '.join(fragments)]))
    assert a._latest_user_text == 'Спасибо.'

    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Запись подтверждена.')

    assert a._pending_outcome is not None
    assert a._pending_outcome['outcome'] == 'agreed'


@pytest.mark.asyncio
@pytest.mark.parametrize('earlier_reply', ['Да', 'Подтверждаю.'])
async def test_prior_turn_yes_cannot_authorize_later_unconfirmed_turn(earlier_reply):
    from livekit.agents.llm import ChatMessage
    a = agent_with_text('')
    a._record_callee_final(earlier_reply)
    await a.on_user_turn_completed(
        a.chat_ctx.copy(), ChatMessage(role='user', content=[earlier_reply]))
    assert a._booking_close_block('agreed') is None
    a._chat_ctx.add_message(role='assistant', content='Запишите, пожалуйста, на имя Армен')
    a._record_callee_final('Спасибо.')
    # Must reject even before the new turn is committed (e.g. during playout).
    assert a._booking_close_block('agreed') is not None
    await a.on_user_turn_completed(
        a.chat_ctx.copy(), ChatMessage(role='user', content=['Спасибо.']))

    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Запись подтверждена.')

    assert a._pending_outcome is None


@pytest.mark.asyncio
@pytest.mark.parametrize('last_fragment', ['Нет, не записывайте.', 'На какое имя?'])
async def test_later_fragment_can_veto_confirmation_in_same_turn(last_fragment):
    a = agent_with_text('')
    a._record_callee_final('Подтверждаю.')
    a._record_callee_final(last_fragment)
    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Запись подтверждена.')
    assert a._pending_outcome is None


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
 'А подскажите пожалуйста вас сколько чс будет.', 'На сколько человек?',
 'Онаопиш.', 'Да, на сколько человек?', 'Нет, мест нет.',
 'Я не подтверждаю.', 'Места есть.', 'Да есть.',
 'Да да получится, если внесёте депозит.', 'Да да не получится.',
])
async def test_no_agreed_outcome_from_question_or_garbled_text(text):
    a=agent_with_text(text)
    result=await g.Gen2BAssistant.finalize_call._func(a,SimpleNamespace(),outcome='agreed',summary='Бронь подтверждена')
    assert a._pending_outcome is None
    assert 'не завершай' in result.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['Да.', 'Да, да, можем.', 'Хорошо, бронируем.', 'Подтверждаю.', 'Записали вас.'])
async def test_one_clear_acceptance_is_enough(text):
    a=agent_with_text(text)
    await g.Gen2BAssistant.finalize_call._func(a,SimpleNamespace(),outcome='agreed',summary='Бронь подтверждена')
    assert a._pending_outcome['outcome']=='agreed'


@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
    'Да могу.',
    'Да, всё нормально, спасибо.',
    'В принципе да, всё нормально, спасибо.',
])
async def test_natural_confirmation_phrases_allow_agreed_outcome(text):
    a=agent_with_text(text)
    await g.Gen2BAssistant.finalize_call._func(a,SimpleNamespace(),outcome='agreed',summary='Запись подтверждена')
    assert a._pending_outcome['outcome']=='agreed'


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['До свидания.', 'Трубку положи.', 'Положите трубку, пожалуйста.'])
async def test_explicit_callee_hangup_request_is_never_blocked(text, monkeypatch):
    a=agent_with_text(text)
    hang=AsyncMock(); monkeypatch.setattr(g,'hang_up_sip_participant',hang)
    session=SimpleNamespace(say=Mock(),shutdown=Mock())

    result=await g.Gen2BAssistant.end_call._func(a,SimpleNamespace(session=session))

    hang.assert_awaited_once_with('guard-test','test')
    session.say.assert_not_called()
    session.shutdown.assert_called_once_with(drain=False)
    assert a._pending_outcome['outcome']=='incomplete'
    assert result=='Звонок завершён.'


@pytest.mark.asyncio
async def test_question_blocks_hangup_even_after_staging(monkeypatch):
    a=agent_with_text('Да.')
    await g.Gen2BAssistant.finalize_call._func(a,SimpleNamespace(),outcome='agreed',summary='Бронь подтверждена')
    a._chat_ctx.add_message(role='user',content='Подождите, на сколько человек?')
    hang=AsyncMock(); monkeypatch.setattr(g,'hang_up_sip_participant',hang)
    shutdown=Mock()
    result=await g.Gen2BAssistant.end_call._func(a,SimpleNamespace(session=SimpleNamespace(shutdown=shutdown)))
    hang.assert_not_awaited();shutdown.assert_not_called()
    assert 'не завершай' in result.lower()


@pytest.mark.asyncio
async def test_booking_command_without_callee_confirmation_does_not_hang_up(monkeypatch):
    a=agent_with_text('Тогда запишите, пожалуйста, завтра в 21:00 к старшему барберу Ануару.')
    a._pending_outcome={'outcome':'agreed'}
    hang=AsyncMock(); monkeypatch.setattr(g,'hang_up_sip_participant',hang)

    result=await g.Gen2BAssistant.end_call._func(a,SimpleNamespace(session=SimpleNamespace(shutdown=Mock())))

    hang.assert_not_awaited()
    assert 'не завершай звонок' in result.lower()
    assert 'нет явного согласия' in result.lower()


@pytest.mark.asyncio
async def test_raw_final_question_arriving_during_playout_blocks_hangup(monkeypatch):
    a=agent_with_text('Да.')
    a._pending_outcome={'outcome':'agreed'}
    a._latest_user_text='На сколько человек?'
    hang=AsyncMock();monkeypatch.setattr(g,'hang_up_sip_participant',hang)
    result=await g.Gen2BAssistant.end_call._func(a,SimpleNamespace(session=SimpleNamespace(shutdown=Mock())))
    hang.assert_not_awaited()
    assert 'не завершай' in result.lower()


@pytest.mark.asyncio
async def test_other_personas_are_unchanged():
    a=agent_with_text('Онаопиш.',enabled=False)
    await g.Gen2BAssistant.finalize_call._func(a,SimpleNamespace(),outcome='agreed',summary='unchanged')
    assert a._pending_outcome is not None


def test_follow_up_keeps_booking_guard_from_prior_call_details():
    import json
    config = g.resolve_call_config(json.dumps({
        'persona': 'armen_personal_assistant',
        'task': 'Есть ли время послезавтра в 21:00?',
        'task_details': 'Исходное поручение: Позвони в ресторан и забронируй стол на 10 человек.',
    }))
    assert config['booking_confirmation_required'] is True


@pytest.mark.asyncio
async def test_production_positive_reply_does_not_force_duplicate_confirmation():
    a = agent_with_text('Да да получится.')
    from livekit.agents.llm import ChatMessage
    a._chat_ctx.items.insert(0, ChatMessage(
        role='assistant', content=['Получится оформить бронь на десять человек завтра в 22:00 '
                                   'с депозитом 50 000 рублей и скидкой 20%?']))
    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Ресторан согласился оформить бронь.')
    assert a._pending_outcome is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('earlier_reply', ['Да.', 'Да есть.', 'Подтверждаю.'])
@pytest.mark.parametrize('post_owner_reply', ['', 'Спасибо.'])
async def test_owner_permission_cannot_reuse_earlier_callee_yes(
        monkeypatch, tmp_path, earlier_reply, post_owner_reply):
    # Follow-up demo_3b91d08158adf0e7: permission to use the name never
    # reached the restaurant. Use an otherwise accepted yes to expose stale evidence.
    a = agent_with_text(earlier_reply)
    a._record_callee_final(earlier_reply)
    a._pending_outcome = {'outcome': 'agreed', 'summary': 'Previous agreement'}
    monkeypatch.setattr(g, 'LIVE_CALLBACK_DIR', tmp_path)
    monkeypatch.setattr(g, 'send_live_callback_to_telegram', lambda *args: 123)
    monkeypatch.setattr(g, 'wait_for_response', AsyncMock(return_value='Оформить на имя Армен'))
    session = SimpleNamespace(say=Mock(), interrupt=AsyncMock(), generate_reply=Mock())
    await g.Gen2BAssistant.ask_owner._func(
        a, SimpleNamespace(session=session),
        question='Разрешаете оформить бронь на имя Армен?')
    await a._owner_continuation_task
    assert a._pending_outcome is None
    a._chat_ctx.add_message(role='user', content='Ответ Армена: Указать только Армен')
    a._record_callee_final(post_owner_reply)
    if post_owner_reply:
        await a.on_user_turn_completed(
            a.chat_ctx.copy(), g.ChatMessage(role='user', content=[post_owner_reply]))
    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Бронь оформлена на имя Армен.')
    assert a._pending_outcome is None
    hang = AsyncMock()
    monkeypatch.setattr(g, 'hang_up_sip_participant', hang)
    result = await g.Gen2BAssistant.end_call._func(a, SimpleNamespace(session=session))
    assert 'не завершай' in result.lower()
    hang.assert_not_awaited()

    # Only actual callee STT may discharge the pending owner decision.
    a._chat_ctx.add_message(role='assistant', content='Оформите, пожалуйста, на имя Армен.')
    a._record_callee_final('Да.')
    await g.Gen2BAssistant.finalize_call._func(
        a, SimpleNamespace(), outcome='agreed', summary='Бронь оформлена на имя Армен.')
    assert a._pending_outcome['outcome'] == 'agreed'
