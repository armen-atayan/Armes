import json

import call_with_persona
import gen2b_agent


def test_dispatch_metadata_carries_web_demo_routing():
    metadata = call_with_persona.build_dispatch_metadata(
        persona="armen_personal_assistant",
        target_name="Антон",
        target_pronoun_nom="он",
        target_pronoun_acc="его",
        target_identity="anton",
        task="Согласовать встречу",
        task_details="Завтра после обеда",
        origin="web_demo",
        owner_channel="web",
        result_channel="web",
        demo_session_id="demo_abc123",
    )

    assert metadata["origin"] == "web_demo"
    assert metadata["owner_channel"] == "web"
    assert metadata["result_channel"] == "web"
    assert metadata["demo_session_id"] == "demo_abc123"


def test_resolved_call_config_preserves_web_demo_routing():
    config = gen2b_agent.resolve_call_config(
        json.dumps(
            {
                "persona": "armen_personal_assistant",
                "target_name": "Антон",
                "task": "Согласовать встречу",
                "origin": "web_demo",
                "owner_channel": "web",
                "result_channel": "web",
                "demo_session_id": "demo_abc123",
            },
            ensure_ascii=False,
        )
    )

    assert config["origin"] == "web_demo"
    assert config["owner_channel"] == "web"
    assert config["result_channel"] == "web"
    assert config["demo_session_id"] == "demo_abc123"
