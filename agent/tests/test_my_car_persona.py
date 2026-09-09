import json
from pathlib import Path


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def load_my_car_persona() -> dict:
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    return personas["my_car"]


def test_my_car_uses_bagdat_voice_and_separated_brand_name():
    persona = load_my_car_persona()
    rendered = " ".join(
        [persona["name"], persona["description"], persona["greeting_instructions"], persona["system_prompt"]]
    )

    assert persona["voice"] == "clone:bagdat"
    assert "Ты Амина, женщина-консультант" in rendered
    assert "Тебя зовут Багдат" not in rendered
    assert "мужчина-консультант" not in rendered
    assert persona["name"] == "My Car"
    assert "Mycar" not in rendered
    assert "My Car" in rendered


def test_my_car_simulates_calendar_conflict_and_books_nearest_slot():
    persona = load_my_car_persona()
    greeting = persona["greeting_instructions"].format(target_name="Армен")
    prompt = persona["system_prompt"].format(target_name="Армен")

    assert "Здравствуйте, это Армен?" not in greeting
    assert "Здравствуйте, это My Car. Вы оставляли заявку на обратный звонок. Подскажите, какой автомобиль вас интересует?" in greeting
    assert "сразу начни разговор" in prompt
    assert "город, удобный день и удобное время" in prompt
    assert "выбранный клиентом слот недоступен" in prompt
    assert "ближайший доступный слот" in prompt
    assert "на один час позже" in prompt
    assert "запись внесена в календарь" in prompt
    assert "менеджер подтвердит визит" not in prompt
    assert "finalize_call" in prompt
    assert "end_call" in prompt


def test_my_car_kazakh_variant_uses_the_same_calendar_simulation():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["my_car_kk"]
    greeting = persona["greeting_instructions"].format(target_name="Эльмира")
    prompt = persona["system_prompt"]

    assert "Сәлеметсіз бе, бұл Эльмира ма?" not in greeting
    assert "Сәлеметсіз бе, бұл My Car. Сіз кері қоңырауға өтінім қалдырған едіңіз. Қай автокөлік сізді қызықтырады?" in greeting

    assert "клиент таңдаған уақыт бос емес" in prompt
    assert "ең жақын бос уақытты" in prompt
    assert "бір сағат кейінгі уақытты" in prompt
    assert "күнтізбеге енгізілгенін" in prompt
    assert "менеджер келуді растайтынын" not in prompt


def test_my_car_simulates_inventory_colors_and_executive_trim():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))

    for key in ("my_car", "my_car_anton"):
        prompt = personas[key]["system_prompt"]
        assert "если клиент спрашивает о наличии" in prompt
        assert "автомобиль есть в наличии" in prompt
        assert "чёрный, синий и белый" in prompt
        assert "комплектация Executive есть в наличии" in prompt
        assert "Не упоминай, что это имитация" in prompt
        assert "не выдумывай точную цену" in prompt

    kazakh_prompt = personas["my_car_kk"]["system_prompt"]
    assert "автокөлік бар екенін айт" in kazakh_prompt
    assert "қара, көк және ақ" in kazakh_prompt
    assert "Executive жинақтамасы бар" in kazakh_prompt


def test_my_car_requires_city_and_budget_before_appointment():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))

    for key in ("my_car", "my_car_anton"):
        prompt = personas[key]["system_prompt"]
        assert "До предложения записи обязательно получи четыре ответа" in prompt
        assert "марка или модель, новый или с пробегом, город и ориентировочный бюджет" in prompt
        assert "Если бюджет не назван, обязательно спроси" in prompt
        assert "Запрещено предлагать запись" in prompt
        assert "Никогда не предполагай город клиента" in prompt
        assert "АстанА" in prompt
        assert "в АстанУ" in prompt
        assert "из АстанЫ" in prompt
        assert "Всегда склоняй название города по контексту" in prompt
        assert "Никогда не используй старое название" in prompt

    kazakh_prompt = personas["my_car_kk"]["system_prompt"]
    assert "Клиенттің қаласын ешқашан өзің болжама" in kazakh_prompt
    assert "АстанАҒА" in kazakh_prompt
    assert "Жазылуды ұсынбас бұрын төрт жауапты міндетті түрде ал" in kazakh_prompt
    assert "Бюджет айтылмаса, міндетті түрде" in kazakh_prompt


def test_my_car_anton_uses_male_voice_and_the_same_sales_flow():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["my_car_anton"]
    greeting = persona["greeting_instructions"].format(target_name="Армен")
    prompt = persona["system_prompt"].format(target_name="Армен")
    rendered = " ".join(str(value) for value in persona.values())

    assert persona["name"] == "My Car — Антон"
    assert persona["voice"] == "clone:anton_m"
    assert "мужчина-консультант" in prompt
    assert "Тебя зовут Антон" in prompt
    assert "Здравствуйте, это Армен?" not in greeting
    assert "Здравствуйте, это My Car. Вы оставляли заявку на обратный звонок. Подскажите, какой автомобиль вас интересует?" in greeting
    assert "выбранный клиентом слот недоступен" in prompt
    assert "ближайший доступный слот" in prompt
    assert "запись внесена в календарь" in prompt
    assert "менеджер подтвердит визит" not in prompt
    assert "Mycar" not in rendered
