from types import SimpleNamespace
from pipeline_diagnostics import attach_pipeline_diagnostics

class Emitter:
    def __init__(self): self.handlers = {}
    def on(self, event, cb): self.handlers[event] = cb

def test_native_provider_and_turn_metrics(caplog):
    stt, llm, tts, session = (Emitter() for _ in range(4))
    attach_pipeline_diagnostics(session, stt, llm, tts, 'diagnostic-only')
    with caplog.at_level('INFO'):
        stt.handlers['metrics_collected'](SimpleNamespace(type='stt_metrics', duration=.6, audio_duration=1.2, secret='not-logged'))
        llm.handlers['metrics_collected'](SimpleNamespace(type='llm_metrics', ttft=2., duration=2.3))
        tts.handlers['metrics_collected'](SimpleNamespace(type='tts_metrics', ttfb=1., duration=1.1))
        session.handlers['conversation_item_added'](SimpleNamespace(item=SimpleNamespace(role='assistant', metrics={'llm_node_ttft':2., 'tts_node_ttfb':1.1, 'provider_request_ids':['safe-id']}, text_content='DO NOT LOG CONTENT')))
    text=caplog.text
    assert text.count('PIPELINE_PROVIDER_METRIC') == 3
    assert 'PIPELINE_TURN_METRIC' in text
    assert '"llm_node_ttft": 2.0' in text
    assert '"duration": 0.6' in text
    assert 'not-logged' not in text and 'DO NOT LOG CONTENT' not in text

def test_handoff_without_metrics_does_not_crash():
    emitters=[Emitter() for _ in range(4)]
    attach_pipeline_diagnostics(*emitters, 'diagnostic-only')
    emitters[0].handlers['conversation_item_added'](SimpleNamespace(item=SimpleNamespace(type='handoff')))
