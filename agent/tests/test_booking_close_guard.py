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
@pytest.mark.parametrize('text',[
 'А подскажите пожалуйста вас сколько чс будет.', 'На сколько человек?',
 'Онаопиш.', 'Да, на сколько человек?', 'Нет, мест нет.',
 'Я не подтверждаю.', 'Места есть.',
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
