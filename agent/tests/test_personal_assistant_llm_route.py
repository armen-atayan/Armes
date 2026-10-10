import json
from pathlib import Path


def test_personal_assistant_uses_gemini_2_5_flash_gateway():
    personas = json.loads(Path(__file__).resolve().parents[1].joinpath('personas.json').read_text())
    assistant = personas['armen_personal_assistant']
    assert assistant['llm_route'] == 'gateway'
    assert assistant['llm_model'] == 'gemini-2.5-flash'
