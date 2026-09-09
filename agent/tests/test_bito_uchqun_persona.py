import json
from pathlib import Path


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def load_persona() -> dict:
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    return personas["bito_uchqun"]


def test_bito_uchqun_identity_language_and_voice():
    persona = load_persona()
    rendered = " ".join(str(value) for value in persona.values())

    assert persona["name"] == "Uchqun — Bito ERP"
    assert persona["voice"] == "clone:uchqun"
    assert persona["tts_speed"] == 1.2
    assert persona["stt_language"] == "uz"
    assert "Bito platformasining CEO" in rendered
    assert "faqat o‘zbek tilida" in rendered
    assert "bir martada faqat bitta savol" in rendered


def test_bito_uchqun_uses_identity_gate_before_sales_pitch():
    persona = load_persona()
    greeting = persona["greeting_instructions"].format(target_name="Aziz")
    prompt = persona["system_prompt"].format(
        target_name="Aziz", target_pronoun_nom="u", target_pronoun_acc="uni"
    )

    assert greeting == "Faqat shuni ayt: «Assalomu alaykum, bu Azizmi?» Keyin jim turib javobni kut."
    assert "shaxsini tasdiqlamaguncha Bito haqida gapirma" in prompt
    assert "noto‘g‘ri raqam" in prompt


def test_bito_uchqun_sales_flow_and_verified_site_facts():
    prompt = load_persona()["system_prompt"]

    for fact in (
        "chakana savdo",
        "onlayn savdo",
        "nasiya savdo",
        "ulgurji savdo",
        "distributsiya",
        "ishlab chiqarish",
        "CRM",
        "ta’minot",
        "HR",
        "logistika",
        "analitika",
        "ombor",
        "markirovka",
        "moliyaviy hisobotlar",
        "300 000 so‘mdan boshlanadi",
        "12 oylik obunaga 2 oy bonus",
        "+998 55 511 37 00",
        "Toshkent shahri, Bunyodkor 40/1",
    ):
        assert fact in prompt

    assert "biznes turi" in prompt
    assert "hozirgi boshqaruv tizimi" in prompt
    assert "asosiy muammo" in prompt
    assert "demo" in prompt
    assert "finalize_call" in prompt
    assert "end_call" in prompt


def test_bito_uchqun_does_not_invent_commercial_terms():
    prompt = load_persona()["system_prompt"]

    assert "aniq narxni o‘ylab topma" in prompt
    assert "Start tarifi" in prompt
    assert "Individual tarif" in prompt
    assert "to‘lov qabul qilma" in prompt
