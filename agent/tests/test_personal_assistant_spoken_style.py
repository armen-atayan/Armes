import json
from pathlib import Path
import gen2b_agent as g


def test_spoken_dates_confirmation_and_acknowledgement_contract():
    p=json.loads((Path(__file__).parents[1]/'personas.json').read_text())['armen_personal_assistant']
    prompt=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','target_name':'сотрудник ресторана','task':'Стол на четверых 9 сентября 2026 года в 20:00.'}))['system_prompt']
    assert 'Короткое подтверждение — только «Хорошо»' in prompt
    assert '«Понятно»' not in p['system_prompt']
    assert 'Уточняющий вопрос ресторана — не согласие на бронь' in prompt
    assert 'Можете, пожалуйста, подтвердить?' in prompt
    assert 'В обычном разговоре о брони называй число, месяц и время без года' in prompt
    assert 'Год произноси только по прямому вопросу собеседника или для устранения реальной неоднозначности' in prompt
    assert 'Год в ПОРУЧЕНИИ — служебная точность' in prompt
    assert 'дословно' not in p['greeting_instructions']
    # Preserve a precise internal date; change speech, not booking data.
    assert '9 сентября 2026 года в 20:00' in prompt
