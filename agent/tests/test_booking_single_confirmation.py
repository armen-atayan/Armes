import json
from pathlib import Path
import gen2b_agent as g


def test_complete_booking_acceptance_does_not_require_second_confirmation():
    prompt=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','target_name':'сотрудник ресторана','task':'Стол на четверых 9 сентября 2026 в 20:00.'}))['system_prompt']
    assert 'Одного явного согласия ресторана на оформление брони достаточно' in prompt
    assert 'повторное подтверждение не запрашивай' in prompt
    assert 'Простое сообщение о наличии мест ещё не подтверждает оформление брони' in prompt
    assert 'затем мягко спроси: «Можете, пожалуйста, подтвердить?»' not in prompt
