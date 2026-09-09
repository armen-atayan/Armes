import json
from pathlib import Path


def test_booking_does_not_solicit_price_or_deposit():
    p = json.loads((Path(__file__).parents[1] / 'personas.json').read_text())['armen_personal_assistant']['system_prompt']
    assert 'Не спрашивай о ценах, стоимости, депозитах, предоплате или минимальном чеке' in p
    assert 'наличие, цену, дату' not in p
    assert 'и итоговую цену либо депозит' not in p
    assert 'Если ресторан сам сообщает об обязательной оплате' in p
