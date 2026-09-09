from pathlib import Path


AGENT_PATH = Path(__file__).resolve().parents[1] / "gen2b_agent.py"


def test_production_krisp_vad_matches_denis_fanout_wiring():
    source = AGENT_PATH.read_text(encoding="utf-8")

    assert "FanoutVAD," in source
    assert "session_vad = FanoutVAD(" in source


def test_production_commits_turns_from_krisp_vad_not_tp():
    source = AGENT_PATH.read_text(encoding="utf-8")

    assert 'turn_detection="vad"' in source


def test_non_english_stt_auto_detects_caller_language():
    source = AGENT_PATH.read_text(encoding="utf-8")

    assert "detect_language=True" in source
