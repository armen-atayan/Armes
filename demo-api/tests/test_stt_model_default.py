from transcription import Gen2BTranscriber


def test_web_transcriber_defaults_to_current_gen2b_stt_model(monkeypatch):
    monkeypatch.delenv("GEN2B_STT_MODEL", raising=False)
    assert Gen2BTranscriber().model == "gen2b/stt"
