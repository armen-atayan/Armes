import importlib.util
import json
from pathlib import Path

import gen2b_agent


_CALL_SCRIPT = Path(__file__).parents[1] / "call_with_persona.py"
_SPEC = importlib.util.spec_from_file_location("call_with_persona", _CALL_SCRIPT)
call_with_persona = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(call_with_persona)


PERSONAS_PATH = Path(__file__).parents[1] / "personas.json"


def test_personal_assistant_persona_contract():
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    persona = personas["armen_personal_assistant"]
    values = {
        "target_name": "ресторан Birch",
        "target_pronoun_nom": "он",
        "target_pronoun_acc": "его",
        "task": "забронировать стол на двоих завтра в 20:00",
        "task_details": "Бронь на имя Армен Атаян; предпочтительно тихий стол.",
    }
    greeting = persona["greeting_instructions"].format(**values)
    prompt = persona["system_prompt"].format(**values)

    assert persona["name"] == "Личный ассистент Армена Атаяна"
    assert persona["voice"] == "clone:armen3"
    assert persona["llm_model"] == "gpt-5.6-luna"
    assert persona["llm_route"] == "default"
    assert persona["stt_language"] == "kk_ru"
    assert persona["wait_for_user_first"] is True
    assert persona["tts_intonation"] is False
    assert persona["sip_trunk_id"] == "ST_N2XKyxsUNuin"
    assert persona["sip_number"] == "74951804257"
    assert "после первой реплики собеседника" in prompt.lower()
    assert "не представляйся личным ассистентом" in prompt.lower()
    assert "личный ассистент армена атаяна" not in greeting.lower()
    assert "обязательно начни с приветствия «здравствуйте»" in prompt.lower()
    assert "после приветствия сразу скажи «подскажите, пожалуйста»" in prompt.lower()
    assert "«здравствуйте, подскажите, пожалуйста…»" in prompt.lower()
    assert "«алло, подскажите, пожалуйста»" not in prompt.lower()
    assert "подскажите, пожалуйста, можно ли забронировать стол" in prompt.lower()
    assert "подскажите, пожалуйста, к вам можно записаться на мужскую стрижку" in prompt.lower()
    assert "подскажите, пожалуйста, у вас есть стол на сегодня в 20:00 на 4 человека?" in prompt.lower()
    assert values["task"] in greeting
    assert values["task"] in prompt
    assert values["task_details"] in prompt
    assert "один вопрос" in prompt.lower()
    assert "говори простыми разговорными словами" in prompt.lower()
    assert "короткое подтверждение — только «хорошо»" in prompt.lower()
    assert "новое поручение прямо во время текущего звонка" in prompt.lower()
    assert "сразу задай собеседнику сам вопрос без пояснений" in prompt.lower()
    assert "только инструмент произносит «секундочку, сейчас уточню»" in prompt.lower()
    assert "не называй текущую дату" in prompt.lower()
    assert "не называй текущее время" in prompt.lower()
    assert "сегодня 13 мая" not in prompt.lower()
    assert "не подтверждай сумму" in prompt.lower()
    assert "переспроси точную сумму" in prompt.lower()
    assert "это возможно" not in prompt.lower()
    assert "это допустимо" not in prompt.lower()
    assert "говори «можно?»" in prompt.lower()
    assert "получится?" in prompt.lower()
    assert "реплика начинается со слова «можно»" in prompt.lower()
    assert "завершай её точкой" in prompt.lower()
    assert "мужчина" in prompt.lower()
    assert "мужском роде" in prompt.lower()
    assert "женский род" in prompt.lower()
    assert "смотрите" not in prompt.lower()
    assert "«понятно»" not in prompt.lower()
    assert "короткое подтверждение — только" in prompt.lower()
    assert "«понял»" not in prompt.lower()
    assert "9:00" in prompt
    assert "21:00" in prompt
    assert "утро или вечер" in prompt.lower()
    assert "ask_owner" in prompt
    assert "секундочку, сейчас уточню" in prompt.lower()
    assert "не завершай звонок" in prompt.lower()
    assert "не выдумывай" in prompt.lower()
    assert "не подтверждай бронирование" in prompt.lower()
    assert "платёж" in prompt.lower()
    assert "finalize_call" in prompt
    assert "end_call" in prompt


def test_resolve_call_config_renders_dynamic_task_fields():
    metadata = json.dumps(
        {
            "persona": "armen_personal_assistant",
            "target_name": "ресторан Birch",
            "target_identity": "birch",
            "task": "забронировать стол на двоих завтра в 20:00",
            "task_details": "Бронь на имя Армен Атаян.",
        },
        ensure_ascii=False,
    )

    config = gen2b_agent.resolve_call_config(metadata)

    assert "забронировать стол на двоих завтра в 20:00" in config["system_prompt"]
    assert "Бронь на имя Армен Атаян." in config["system_prompt"]
    assert config["task"] == "забронировать стол на двоих завтра в 20:00"
    assert config["task_details"] == "Бронь на имя Армен Атаян."
    assert config["tts_intonation"] is False


def test_dispatch_metadata_includes_task_and_details():
    metadata = call_with_persona.build_dispatch_metadata(
        persona="armen_personal_assistant",
        target_name="ресторан Birch",
        target_pronoun_nom="он",
        target_pronoun_acc="его",
        target_identity="birch",
        task="забронировать стол на двоих завтра в 20:00",
        task_details="Бронь на имя Армен Атаян.",
    )

    assert metadata["task"] == "забронировать стол на двоих завтра в 20:00"
    assert metadata["task_details"] == "Бронь на имя Армен Атаян."


def test_persona_trunk_is_selected_before_global_default():
    persona = {"sip_trunk_id": "ST_persona"}

    assert call_with_persona.resolve_trunk_id("", persona, "ST_global") == "ST_persona"
    assert call_with_persona.resolve_trunk_id("ST_explicit", persona, "ST_global") == "ST_explicit"
    assert call_with_persona.resolve_sip_number("", {"sip_number": "74951804257"}) == "74951804257"
    assert call_with_persona.resolve_sip_number("78005553535", {"sip_number": "74951804257"}) == "78005553535"


def test_unknown_guest_count_waits_for_callee_to_require_it():
    config = gen2b_agent.resolve_call_config(
        json.dumps(
            {
                "persona": "armen_personal_assistant",
                "target_name": "ресторан Birch",
                "task": "забронировать стол 9 сентября в 20:00",
                "task_details": "",
            },
            ensure_ascii=False,
        )
    )

    prompt = config["system_prompt"].lower()
    assert "не вызывай ask_owner заранее только потому, что параметр отсутствует" in prompt
    assert "сначала озвучь основной запрос" in prompt
    assert "собеседник явно запросил этот параметр" in prompt
    assert "не произноси никакое число" in prompt
    assert "нельзя угадывать" in prompt


def test_missing_owner_booking_details_must_not_be_requested_from_callee():
    config = gen2b_agent.resolve_call_config(
        json.dumps(
            {
                "persona": "armen_personal_assistant",
                "target_name": "ресторан",
                "task": "Позвони и забронируй стол, остальное уточню",
                "task_details": "",
            },
            ensure_ascii=False,
        )
    )

    prompt = config["system_prompt"].lower()
    assert "дату и время брони определяет армен" in prompt
    assert "не спрашивай у собеседника, на какую дату или время нужна бронь" in prompt
    assert "сначала вызови ask_owner" in prompt
