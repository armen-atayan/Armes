import json
from pathlib import Path


def test_personal_assistant_uses_native_haiku_5_5_route():
    personas = json.loads(Path(__file__).resolve().parents[1].joinpath('personas.json').read_text())
    assistant = personas['armen_personal_assistant']
    assert assistant['llm_route'] == 'anthropic'
    assert assistant['llm_model'] == 'claude-haiku-5-5'
