import json
from pathlib import Path


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def test_vinyl_interest_persona_is_truthful_and_non_transactional():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["vinyl_interest_research"]
    greeting = persona["greeting_instructions"].format(target_name="Леонид")
    prompt = persona["system_prompt"].format(
        target_name="Леонид", target_pronoun_nom="он", target_pronoun_acc="его"
    )

    assert persona["voice"] == "clone:anton_m"
    assert "Алло, это Леонид?" in greeting
    lowered = prompt.lower()
    assert "товара на руках нет" in lowered
    assert "оплату мы не принимаем" in lowered
    assert "по просьбе армена" in lowered
    assert "Beatles" in prompt
    assert "20 000" in prompt
    assert "гипотетическую" in prompt
    assert "не выдавай" in lowered
    assert "finalize_call" in prompt
    assert "end_call" in prompt
