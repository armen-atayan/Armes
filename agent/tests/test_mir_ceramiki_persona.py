import json
from pathlib import Path


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def test_mir_ceramiki_male_callback_persona_contract():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["mir_ceramiki_anton"]
    greeting = persona["greeting_instructions"].format(
        target_name="клиент", target_pronoun_nom="он", target_pronoun_acc="его"
    )
    prompt = persona["system_prompt"].format(
        target_name="клиент", target_pronoun_nom="он", target_pronoun_acc="его"
    )

    assert persona["name"] == "Антон — Мир Керамики"
    assert persona["voice"] == "clone:anton_m"
    assert persona["llm_model"] == "gpt-5.6-luna"
    assert persona["stt_language"] == "kk_ru"
    assert "пропущенный звонок" in greeting.lower()
    assert "Мир Керамики" in greeting
    assert "плитки" in greeting.lower()
    assert "сантехники" in greeting.lower()

    lowered = prompt.lower()
    for fact in (
        "керамогранит",
        "керамическая плитка",
        "сантехника",
        "мебель для ванной",
        "ламинат",
        "бесплатный 3d-дизайн",
        "23 года",
    ):
        assert fact in lowered

    assert "один вопрос" in lowered
    assert "проактив" in lowered
    assert "офис продаж" in lowered
    assert "предлагай визит" in lowered
    assert "день" in lowered
    assert "время" in lowered
    assert "явно подтверд" in lowered
    assert "не выдумывай" in lowered
    assert "город" in lowered
    assert "наличие" in lowered
    assert "finalize_call" in prompt
    assert "end_call" in prompt
