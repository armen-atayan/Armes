import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'probe_pipeline_latency.py'
spec = importlib.util.spec_from_file_location('latency_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def pcm(ms, amplitude=0, rate=8000):
    return struct.pack('<h', amplitude) * (rate * ms // 1000)


def test_raw_provider_and_modern_turn_metrics_are_room_scoped():
    room = 'diagnostic-one'
    rows = [
        {'MESSAGE': json.dumps({'room': room, 'message': 'PIPELINE_PROVIDER_METRIC room=diagnostic-one module=llm_metrics metrics={"ttft": 2.1, "duration": 2.3}', 'timestamp': '2026-10-09T12:00:00+00:00'})},
        {'MESSAGE': json.dumps({'room': room, 'message': 'PIPELINE_TURN_METRIC room=diagnostic-one role=user metrics={"transcription_delay": 0.7, "end_of_turn_delay": 0.9}'})},
        {'MESSAGE': json.dumps({'room': 'other', 'message': 'PIPELINE_PROVIDER_METRIC room=other module=llm_metrics metrics={"ttft": 99}'})},
    ]
    got = probe.parse_detailed_journal_metrics('\n'.join(map(json.dumps, rows)), room)
    assert got['providers'][0]['metrics']['ttft'] == 2.1
    assert got['turns'][0]['metrics']['end_of_turn_delay'] == .9
    assert len(got['providers']) == 1


def test_statistics_nearest_rank():
    assert probe.stats([]) == {'count': 0, 'min': None, 'median': None, 'p95': None}
    assert probe.stats([7]) == {'count': 1, 'min': 7, 'median': 7, 'p95': 7}
    assert probe.stats(range(1, 21)) == {'count': 20, 'min': 1, 'median': 10.5, 'p95': 19}
    assert probe.stats([1, 2, 3])['p95'] == 3


def test_input_end_uses_last_speech_block_not_clip_end():
    result = probe.input_evidence(pcm(100) + pcm(60, 1000) + pcm(100))
    assert result['speech_end_offset_s'] == pytest.approx(.16)
    assert result['threshold_uncertainty_ms'] == 20
    assert result['frame_bytes'] == 320
    assert result['frames'][5]['rms_dbfs'] == pytest.approx(20 * math.log10(1000 / 32768))
    with pytest.raises(ValueError, match='speech'):
        probe.input_evidence(pcm(100))


def test_output_rejects_pulse_and_reports_onset_and_offset():
    detector = probe.OutputDetector(8000)
    raw = pcm(20, 1000) + pcm(40) + pcm(60, 1000) + pcm(2000)
    # Arbitrary decoder packet boundaries must not change the acoustic test.
    for pos in range(0, len(raw), 160):
        detector.feed(raw[pos:pos + 160], 10 + (pos + 160) / 16000)
    assert detector.onset_s == pytest.approx(.06)
    assert detector.onset_monotonic == pytest.approx(10.08)
    assert detector.confirmed_monotonic == pytest.approx(10.10)
    assert detector.last_speech_end_s == pytest.approx(.12)
    assert detector.finished


def test_journal_exact_room_and_job_pid_and_missing():
    def entry(body, **outer):
        return json.dumps({'MESSAGE': json.dumps(body), **outer})
    summary = 'LATENCY_SUMMARY module=tts_ttfb count=2 mean_ms=30.000 p95_ms=20.000 max_ms=40.000'
    lines = [
        entry({'message': 'job started', 'room': 'diagnostic-one', 'pid': 71}),
        entry({'message': summary, 'pid': 71}),
        entry({'message': summary, 'room': 'diagnostic-one-extra', 'pid': 72}),
        entry({'message': summary, 'room': 'other', 'pid': 72}),
        entry({'message': summary, 'pid': 99}),
        entry({'message': summary + ' diagnostic-one'}),
        'invalid json',
    ]
    result = probe.parse_journal('\n'.join(lines), 'diagnostic-one')
    assert len(result['tts_ttfb']['summaries']) == 1
    assert result['tts_ttfb']['summaries'][0]['mean_ms'] == 30
    assert result['tts_ttfb']['first_request_ms'] is None
    assert result['stt_request']['status'] == 'missing'
    assert result['tts_ttfb']['summaries'][0]['p95_method'] == 'worker_reported_not_recomputed'


def test_import_has_no_effects():
    code = '''
import importlib.util, subprocess, os, socket, asyncio
from pathlib import Path
def forbidden(*a, **k): raise AssertionError('import side effect')
subprocess.Popen = forbidden
socket.socket = forbidden
os.putenv = forbidden
Path.read_bytes = forbidden
spec = importlib.util.spec_from_file_location('probe', %r)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert 'livekit' not in __import__('sys').modules
''' % str(PATH)
    subprocess.run([sys.executable, '-c', code], check=True)


def test_metadata_routes_web_and_unique_session():
    metadata = probe.dispatch_metadata('diagnostic-one')
    assert metadata['persona'] == 'armen_personal_assistant'
    assert metadata['origin'] == 'web_demo'
    assert metadata['owner_channel'] == metadata['result_channel'] == 'web'
    assert metadata['demo_session_id'] == 'diagnostic-one'


def test_ambiguous_job_pid_cannot_attribute_summary():
    records = [json.dumps({'MESSAGE': json.dumps(row)}) for row in [
        {'message': 'start', 'room': 'diagnostic-one', 'pid': 71},
        {'message': 'start', 'room': 'other', 'pid': 71},
        {'message': 'LATENCY_SUMMARY module=llm_ttft count=1 mean_ms=3 p95_ms=3 max_ms=3', 'pid': 71},
    ]]
    assert probe.parse_journal('\n'.join(records), 'diagnostic-one')['llm_ttft']['status'] == 'missing'


def test_structured_room_fields_override_journal_parent_pid():
    line = json.dumps({'_PID': '99', 'MESSAGE': json.dumps({
        'context': {'room_name': 'diagnostic-one'},
        'message': 'LATENCY_SUMMARY module=llm_ttft count=1 mean_ms=3 p95_ms=3 max_ms=3'})})
    assert probe.parse_journal(line, 'diagnostic-one')['llm_ttft']['status'] == 'reported'


def test_measured_chain_uses_decoder_receipt_not_backdated_packet_start():
    detector = probe.OutputDetector(8000)
    detector.feed(pcm(40, 1000), 12.0)
    assert detector.onset_monotonic == 12.0
    assert detector.onset_s == 0


def test_failure_closes_only_diagnostic_room_and_saves_evidence(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace

    calls = []

    class Room:
        def on(self, name):
            return lambda fn: fn

        async def connect(self, *args):
            raise TimeoutError('must not be copied to evidence')

        async def disconnect(self):
            calls.append('disconnect')

    class Token:
        def __init__(self, *args):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

        def to_jwt(self):
            return 'fake-token'

    class API:
        def __init__(self, **kwargs):
            self.room = self

        async def delete_room(self, request):
            calls.append(('delete', request.room))

        async def list_rooms(self, request):
            calls.append(('list', request.names))
            return SimpleNamespace(rooms=[])

        async def aclose(self):
            calls.append('api-close')

    fake = SimpleNamespace(
        rtc=SimpleNamespace(Room=Room),
        api=SimpleNamespace(LiveKitAPI=API, AccessToken=Token, VideoGrants=SimpleNamespace,
                            DeleteRoomRequest=SimpleNamespace, ListRoomsRequest=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'livekit', fake)

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(probe.asyncio, 'sleep', no_sleep)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *args, **kwargs: '')
    args = SimpleNamespace(output_dir=tmp_path)
    raw = pcm(40, 1000)
    result = asyncio.run(probe.trial(args, raw, probe.input_evidence(raw),
                                    {'LIVEKIT_API_KEY': 'fake', 'LIVEKIT_API_SECRET': 'fake'}))
    assert result['status'] == 'error'
    assert result['error_type'] == 'TimeoutError'
    assert result['cleanup_errors'] == []
    assert result['room_leak'] is False
    assert result['capture_tasks_pending'] == 0
    assert result['room'].startswith('diagnostic-pipeline-')
    assert calls == ['disconnect', ('delete', result['room']), ('list', [result['room']]), 'api-close']
    assert Path(result['received_wav']).exists()
    assert all(v['status'] == 'missing' for v in result['worker_metrics'].values())
    assert 'must not be copied' not in json.dumps(result)
