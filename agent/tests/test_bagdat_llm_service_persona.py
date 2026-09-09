import json
from pathlib import Path


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def test_all_personas_use_latest_available_gemini_flash_low_route():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))

    assert personas
    assert {persona["llm_model"] for persona in personas.values()} == {"gpt-5.6-luna"}
    assert {persona["llm_route"] for persona in personas.values()} == {"default"}


def test_bagdat_llm_service_persona_contract():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["bagdat_llm_service"]
    greeting = persona["greeting_instructions"].format(target_name="Дима")
    prompt = persona["system_prompt"].format(
        target_name="Дима", target_pronoun_nom="он", target_pronoun_acc="его"
    )

    assert persona["voice"] == "clone:bagdat"
    assert persona["llm_model"] == "gpt-5.6-luna"
    assert persona["llm_route"] == "default"
    assert persona["stt_language"] == "kk_ru"
    assert persona["wait_for_user_first"] is True
    assert "Привет, Дима, это Багдат" in greeting
    assert "Салем" not in greeting
    assert "Алло, это Дима?" not in greeting
    assert "Это Дима?" not in prompt
    assert "первым ничего не говори" in prompt
    assert "Не задавай вопросов о личности" in prompt
    exact_opener = (
        "Привет, Дима, это Багдат. Вчера Айбек показал свои продукты, и там мне "
        "всё понравилось. Но он не показал продукты LLM as a Service. В чём проблема?"
    )
    assert exact_opener in greeting
    assert exact_opener in prompt
    assert "только на «ты»" in prompt
    assert "конкретный" in prompt
    assert "finalize_call" in prompt
    assert "end_call" in prompt
    assert "не оскорбляй" in prompt.lower()
