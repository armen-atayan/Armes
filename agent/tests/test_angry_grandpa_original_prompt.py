import json
from pathlib import Path

import gen2b_agent as g


ORIGINAL_PROMPT_PATH = Path("/home/ubuntu/original_angry_grandpa.txt")


def test_angry_grandpa_uses_exact_original_prompt_file_without_shared_layers():
    config = g.resolve_call_config(
        json.dumps(
            {
                "persona": "angry_grandpa",
                "target_name": "Армен",
                "target_pronoun_nom": "он",
                "target_pronoun_acc": "его",
            },
            ensure_ascii=False,
        )
    )

    expected = ORIGINAL_PROMPT_PATH.read_text(encoding="utf-8").strip().format(
        target_name="Армен",
        target_pronoun_nom="он",
        target_pronoun_acc="его",
    )
    assert config["system_prompt"] == expected
    assert "умеренный мат" not in config["system_prompt"]
    assert "УНИВЕРСАЛЬНОЕ УТОЧНЕНИЕ" not in config["system_prompt"]
