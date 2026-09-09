import json
from pathlib import Path

import gen2b_agent as g


ROOT = Path(__file__).parents[1]


def rendered(task="Запиши на массаж завтра в 20:00.", details=""):
    return g.resolve_call_config(json.dumps({
        "persona": "armen_personal_assistant",
        "target_name": "массажный салон",
        "task": task,
        "task_details": details,
    }, ensure_ascii=False))


def test_hybrid_prompt_is_loaded_from_a_dedicated_document_without_shared_tone_appendix():
    persona = json.loads((ROOT / "personas.json").read_text(encoding="utf-8"))["armen_personal_assistant"]
    config = rendered()
    prompt = config["system_prompt"]

    assert persona["system_prompt_path"].endswith("prompts/armen_personal_assistant_system.txt")
    assert persona["exact_system_prompt"] is True
    assert prompt.startswith("РОЛЬ И ПОЛНОМОЧИЯ")
    assert "## Глобальный стиль живой речи" not in prompt
    assert "УНИВЕРСАЛЬНОЕ УТОЧНЕНИЕ — инструмент" not in prompt


def test_hybrid_prompt_renders_call_data_and_keeps_examples_non_authoritative():
    config = rendered("Запиши на тайский массаж завтра в 20:00.", "Мастер — Ирина.")
    prompt = config["system_prompt"]

    assert "<call_data>" in prompt
    assert "ПОРУЧЕНИЕ: Запиши на тайский массаж завтра в 20:00." in prompt
    assert "ДЕТАЛИ И ОГРАНИЧЕНИЯ: Мастер — Ирина." in prompt
    assert "Не бери факты из примеров" in prompt
    assert "<examples>" in prompt


def test_hybrid_prompt_blocks_inferred_single_visitor_and_defers_unknown_count():
    prompt = rendered()["system_prompt"]

    assert "не означают одного посетителя" in prompt
    assert "Количество посетителей известно только" in prompt
    assert "единственное действие текущего ответа: вызов ask_owner" in prompt
    assert "не собирай недостающие параметры до первого запроса о доступности" in prompt.lower()


def test_hybrid_prompt_prioritizes_callee_clarification_for_ambiguous_money():
    prompt = rendered()["system_prompt"]

    assert "не вызывай ask_owner" in prompt
    assert "сначала попроси организацию повторить точную сумму, валюту" in prompt
    assert "Только после получения однозначных условий" in prompt


def test_hybrid_prompt_matches_runtime_farewell_and_first_turn_identity_behavior():
    config = rendered()
    prompt = config["system_prompt"]
    greeting = config["greeting_instructions"]

    assert "Не произноси прощание самостоятельно" in prompt
    assert "его воспроизводит end_call" in prompt
    assert "Если первая реплика содержит прямой содержательный вопрос" in greeting
    assert "Я голосовой ассистент Армена" in prompt
    assert "Приветствие произнеси один раз за звонок" in greeting


def test_hybrid_ask_owner_description_matches_runtime_and_free_answer_buttons():
    description = g.Gen2BAssistant.ask_owner._func.__doc__

    assert "Не сопровождай его речью" in description
    assert "Нажми «Другое»" in description
    assert '["Отложить", "Не продолжать"]' in description
    assert "Telegram показывает универсальные кнопки" in description
    assert "В web-интерфейсе options" in description
