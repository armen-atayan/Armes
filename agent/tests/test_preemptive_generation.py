from pathlib import Path


AGENT_PATH = Path(__file__).resolve().parents[1] / "gen2b_agent.py"


def test_krisp_turn_handling_enables_preemptive_llm_and_tts():
    source = AGENT_PATH.read_text(encoding="utf-8")

    assert 'preemptive_generation=PreemptiveGenerationOptions(' in source
    assert 'enabled=True' in source
    assert 'preemptive_tts=True' in source
