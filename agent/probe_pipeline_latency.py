"""Opt-in production-worker room replay; never invokes SIP. Import is inert.

Example (run only after review)::
    agent/venv/bin/python agent/probe_pipeline_latency.py --input recording.ogg \
        --start 8.5 --duration 2.2 --repeats 5 --output-dir /tmp/pipeline-latency

Client latency includes LiveKit transport but excludes SIP and setup. Acoustic
boundaries are threshold estimates, not semantic turn-end annotations. Worker
p95 values are preserved verbatim: raw requests are unavailable for recomputing.
"""
import argparse
import asyncio
import inspect
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess
import time
import uuid
import wave

MODULES = ('stt_request', 'llm_ttft', 'llm_total', 'tts_ttfb', 'tts_total',
           'eou_delay', 'transcription_delay')
RATE = 8000
FRAME_BYTES = 320
UNIT = 'gen2b-agent.service'


def stats(values):
    values = sorted(values)
    return {'count': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'p95': values[math.ceil(.95 * len(values)) - 1] if values else None}


def rms_dbfs(raw):
    samples = struct.unpack('<' + 'h' * (len(raw) // 2), raw)
    power = sum(x * x for x in samples) / len(samples) if samples else 0
    return 10 * math.log10(power / 32768 ** 2) if power else -120.0


def input_evidence(raw):
    frames = []
    end = None
    for pos in range(0, len(raw), FRAME_BYTES):
        block = raw[pos:pos + FRAME_BYTES]
        db = rms_dbfs(block)
        frames.append({'offset_s': pos / 16000, 'bytes': len(block), 'rms_dbfs': db})
        if db >= -40:
            end = (pos + len(block)) / 16000
    if end is None:
        raise ValueError('No input speech above -40 dBFS')
    return {'speech_end_offset_s': end, 'threshold_dbfs': -40,
            'threshold_uncertainty_ms': 20, 'frame_bytes': FRAME_BYTES,
            'sample_rate': RATE, 'channels': 1, 'pcm_bytes': len(raw), 'frames': frames}


class OutputDetector:
    """Reblock decoded PCM; retain actual receipt time of the first speech block."""
    def __init__(self, rate=24000):
        self.rate = rate
        self.block_bytes = rate // 50 * 2
        self.pending = b''
        self.blocks = 0
        self.run = 0
        self.onset_s = None
        self.onset_monotonic = None
        self.confirmed_monotonic = None
        self.last_speech_end_s = None
        self.finished = False
        self.frames = []
        self.candidate_monotonic = None

    def feed(self, raw, received_monotonic):
        self.pending += raw
        while len(self.pending) >= self.block_bytes:
            block, self.pending = self.pending[:self.block_bytes], self.pending[self.block_bytes:]
            offset = self.blocks * .02
            db = rms_dbfs(block)
            self.frames.append({'offset_s': offset, 'bytes': len(block), 'rms_dbfs': db,
                                'received_monotonic': received_monotonic})
            if db >= -45:
                if not self.run:
                    self.candidate_monotonic = received_monotonic
                self.run += 1
                if self.run >= 2 and self.onset_s is None:
                    self.onset_s = offset - .02
                    self.onset_monotonic = self.candidate_monotonic
                    self.confirmed_monotonic = received_monotonic
                if self.onset_s is not None:
                    self.last_speech_end_s = offset + .02
            else:
                self.run = 0
            self.blocks += 1
            if self.last_speech_end_s is not None:
                self.finished = self.blocks * .02 - self.last_speech_end_s >= 2 - 1e-9


def dispatch_metadata(name):
    return {'persona': 'armen_personal_assistant', 'origin': 'web_demo',
            'owner_channel': 'web', 'result_channel': 'web', 'demo_session_id': name,
            'target_name': 'диагностический стенд без телефонного звонка',
            'task': 'Техническая проверка голосового конвейера. Дождись первой реплики собеседника согласно текущему промпту, затем спроси, хорошо ли тебя слышно. Не выполняй бронирования, заказы, звонки, внешние действия и вызовы инструментов.'}


def parse_journal(raw, room_name):
    records = []
    for line in raw.splitlines():
        try:
            outer = json.loads(line)
            body = outer.get('MESSAGE', {})
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = {'message': body}
            if not isinstance(body, dict):
                continue
            fields = dict(outer)
            fields.update(body)
            for key in ('extra', 'context'):
                if isinstance(body.get(key), dict):
                    fields.update(body[key])
            rooms = [fields[k] for k in ('room', 'room_name', 'roomName') if k in fields]
            rooms = [r.get('name') if isinstance(r, dict) else r for r in rooms]
            # _PID is the journal emitter (may be parent worker), not job pid.
            pid = fields.get('job_pid', fields.get('pid'))
            records.append((rooms, str(pid) if pid is not None else None,
                            fields.get('message', fields.get('msg', ''))))
        except (ValueError, TypeError, AttributeError):
            continue
    pids = {pid for rooms, pid, _ in records if rooms and all(r == room_name for r in rooms) and pid}
    # A reused/ambiguous PID must never admit another room's metrics.
    pids -= {pid for rooms, pid, _ in records if rooms and room_name not in rooms}
    result = {m: {'status': 'missing', 'summaries': [], 'first_request_ms': None} for m in MODULES}
    pattern = re.compile(r'LATENCY_SUMMARY module=(\w+) count=(\d+) mean_ms=([\d.]+) p95_ms=([\d.]+) max_ms=([\d.]+)')
    for rooms, pid, message in records:
        if not ((rooms and all(r == room_name for r in rooms)) or (not rooms and pid in pids)):
            continue
        match = pattern.search(str(message))
        if match and match[1] in result:
            item = result[match[1]]
            item['status'] = 'reported'
            item['summaries'].append({'count': int(match[2]), 'mean_ms': float(match[3]),
                                      'p95_ms': float(match[4]), 'max_ms': float(match[5]),
                                      'p95_method': 'worker_reported_not_recomputed'})
    return result


def parse_detailed_journal_metrics(raw, room_name):
    result = {'providers': [], 'turns': []}
    for line in raw.splitlines():
        try:
            body = json.loads(line).get('MESSAGE', {})
            if isinstance(body, str):
                body = json.loads(body)
            if body.get('room') != room_name:
                continue
            message = body.get('message', '')
            match = re.fullmatch(r'PIPELINE_(PROVIDER|TURN)_METRIC room=' + re.escape(room_name) + r' (module|role)=(\w+) metrics=(\{.*\})', message)
            if match:
                result['providers' if match[1] == 'PROVIDER' else 'turns'].append({
                    match[2]: match[3], 'timestamp': body.get('timestamp'),
                    'metrics': json.loads(match[4])})
        except (ValueError, TypeError, AttributeError):
            continue
    return result


def worker_environment():
    pid = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
                                   '-p', 'MainPID', '--value'], text=True).strip()
    if not pid.isdigit() or int(pid) <= 0:
        raise RuntimeError('Production worker is not running')
    env = dict(x.decode().split('=', 1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x)
    return env


async def trial(args, raw, evidence, env):
    from livekit import api, rtc
    name = 'diagnostic-pipeline-' + uuid.uuid4().hex
    result = {'room': name, 'status': 'starting', 'input': evidence, 'transcripts': [],
              'scope': 'client end-of-speech to decoded non-silent audio; LiveKit, without SIP',
              'output_threshold_dbfs': -45, 'output_sustain_ms': 40,
              'output_timing': 'Actual decoder callback receipt of first above-threshold 20ms block, accepted after 40ms sustained; PCM onset offset recorded separately',
              'input_timing': 'Actual frame submission plus acoustic block end offset; source queue and transport are included in client chain',
              'tts_note': 'Worker summaries aggregate requests; mean is not first sentence/request latency.'}
    started_wall = time.time()
    result['trial_start_unix'] = started_wall
    origin = time.monotonic()
    room = rtc.Room()
    client = api.LiveKitAPI(url='http://localhost:7880', api_key=env['LIVEKIT_API_KEY'], api_secret=env['LIVEKIT_API_SECRET'])
    source = None
    tasks = []
    streams = []
    chunks = []
    detector = OutputDetector()
    ready = asyncio.Event()
    done = asyncio.Event()
    subscribed_identity = None
    input_end = None
    cleanup_errors = []

    async def capture(track):
        stream = rtc.AudioStream(track, sample_rate=24000, num_channels=1, frame_size_ms=20)
        streams.append(stream)
        result['audio_ready_monotonic'] = time.monotonic()
        ready.set()
        async for event in stream:
            now = time.monotonic()
            data = bytes(event.frame.data)
            chunks.append(data)
            detector.feed(data, now)
            if detector.finished:
                done.set()

    @room.on('track_subscribed')
    def on_track(track, publication, participant):
        nonlocal subscribed_identity
        if track.kind == rtc.TrackKind.KIND_AUDIO and subscribed_identity is None:
            subscribed_identity = participant.identity
            result['subscribed_monotonic'] = time.monotonic()
            tasks.append(asyncio.create_task(capture(track)))

    @room.on('transcription_received')
    def on_text(segments, participant, publication):
        now = time.monotonic()
        for segment in segments:
            result['transcripts'].append({'monotonic': now, 'trial_offset_s': now - origin,
                                         'identity': participant.identity if participant else '',
                                         'text': segment.text, 'final': segment.final,
                                         'segment_id': segment.id})

    try:
        token = (api.AccessToken(env['LIVEKIT_API_KEY'], env['LIVEKIT_API_SECRET'])
                 .with_identity('diagnostic-callee-' + uuid.uuid4().hex)
                 .with_grants(api.VideoGrants(room_join=True, room=name))
                 .with_attributes({'sip.callStatus': 'active'}).to_jwt())
        await asyncio.wait_for(room.connect('ws://localhost:7880', token), 20)
        kwargs = {'queue_size_ms': 20} if 'queue_size_ms' in inspect.signature(rtc.AudioSource).parameters else {}
        source = rtc.AudioSource(8000, 1, **kwargs)
        result['source_queue_size_ms'] = kwargs.get('queue_size_ms')
        track = rtc.LocalAudioTrack.create_audio_track('original-recording', source)
        await asyncio.wait_for(room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)), 20)
        await asyncio.wait_for(client.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
            agent_name='gen2b-agent', room=name, metadata=json.dumps(dispatch_metadata(name), ensure_ascii=False))), 20)
        await asyncio.wait_for(ready.wait(), 20)
        # Stream constructed and subscription acknowledged before pre-roll/input.
        start = time.monotonic()
        result['capture_start_monotonic'] = start
        payload = bytes(16000) + raw
        speech_end_bytes = 16000 + round(evidence['speech_end_offset_s'] * 16000)
        result['send_frames'] = []
        index = 0
        while True:
            deadline = start + index * .02
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            pos = index * FRAME_BYTES
            block = payload[pos:pos + FRAME_BYTES].ljust(FRAME_BYTES, b'\0')
            sent = time.monotonic()
            await asyncio.wait_for(source.capture_frame(rtc.AudioFrame(block, 8000, 1, 160)), 2)
            captured = time.monotonic()
            result['send_frames'].append({'index': index, 'monotonic': sent,
                                          'capture_completed_monotonic': captured,
                                          'capture_wait_ms': (captured - sent) * 1000,
                                          'source_queued_s': getattr(source, 'queued_duration', None),
                                          'late_ms': (sent - deadline) * 1000, 'bytes': len(block)})
            if pos < speech_end_bytes <= pos + FRAME_BYTES:
                input_end = sent + (speech_end_bytes - pos) / 16000
                result['input_speech_end_monotonic'] = input_end
            index += 1
            if input_end is not None and time.monotonic() >= input_end + 30:
                result['status'] = 'reply_timeout'
                break
            if pos + FRAME_BYTES >= len(payload) and done.is_set():
                result['status'] = 'complete'
                break
            for task in tasks:
                if task.done():
                    task.result()
                    raise RuntimeError('Output stream ended before capture completed')
        await asyncio.wait_for(source.wait_for_playout(), 2)
        onset = detector.onset_monotonic
        result['first_nonsilent_monotonic'] = onset
        result['onset_confirmed_monotonic'] = detector.confirmed_monotonic
        if onset is not None and input_end is not None and onset >= input_end:
            result['latency_ms'] = (onset - input_end) * 1000
        else:
            result['latency_ms'] = None
            result['latency_missing_reason'] = 'no speech or output began before input speech ended (callee-first violation/overlap)'
    except Exception as exc:
        result['status'] = 'error'
        result['error_type'] = type(exc).__name__  # Exception text may contain credentials.
    finally:
        async def close(label, awaitable):
            try:
                await asyncio.wait_for(awaitable, 10)
            except Exception as exc:
                cleanup_errors.append({'resource': label, 'error_type': type(exc).__name__})
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in streams:
            await close('audio_stream', stream.aclose())
        if source is not None:
            await close('audio_source', source.aclose())
        await close('rtc_room', room.disconnect())
        await close('diagnostic_room_delete', client.room.delete_room(api.DeleteRoomRequest(room=name)))
        try:
            remaining = await asyncio.wait_for(client.room.list_rooms(api.ListRoomsRequest(names=[name])), 10)
            result['room_leak'] = any(r.name == name for r in remaining.rooms)
        except Exception as exc:
            result['room_leak'] = None
            cleanup_errors.append({'resource': 'room_leak_check', 'error_type': type(exc).__name__})
        await close('api', client.aclose())
        result['capture_tasks_pending'] = sum(not task.done() for task in tasks)
        result['cleanup_errors'] = cleanup_errors
    # Allow bounded journal flush after room close; never broaden the room filter.
    await asyncio.sleep(2)
    ended_wall = time.time()
    result['trial_end_unix'] = ended_wall
    journal_cmd = ['journalctl', '--user', '-u', UNIT, '--since', '@' + str(started_wall),
                   '--until', '@' + str(ended_wall), '-o', 'json', '--no-pager']
    if env.get('INVOCATION_ID'):
        journal_cmd.append('_SYSTEMD_INVOCATION_ID=' + env['INVOCATION_ID'])
    try:
        journal = await asyncio.to_thread(subprocess.check_output, journal_cmd, text=True, timeout=10, stderr=subprocess.DEVNULL)
        result['worker_metrics'] = parse_journal(journal, name)
        result['detailed_metrics'] = parse_detailed_journal_metrics(journal, name)
    except Exception as exc:
        result['worker_metrics'] = parse_journal('', name)
        result['journal_error_type'] = type(exc).__name__
    wav_path = args.output_dir / (name + '.wav')
    with wave.open(str(wav_path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b''.join(chunks))
    result['received_wav'] = str(wav_path)
    result['output_pcm'] = {'sample_rate': 24000, 'channels': 1, 'bytes': sum(map(len, chunks)),
                            'frames': detector.frames, 'onset_offset_s': detector.onset_s,
                            'last_speech_end_offset_s': detector.last_speech_end_s,
                            'trailing_partial_bytes': len(detector.pending)}
    return result


async def run(args):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(args.start), '-t', str(args.duration),
                                   '-i', str(args.input), '-f', 's16le', '-ac', '1', '-ar', '8000', '-'],
                                  stderr=subprocess.DEVNULL, timeout=60)
    evidence = input_evidence(raw)
    env = worker_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'derived-input.pcm').write_bytes(raw)
    results = {'input_original': str(args.input.resolve()), 'start_s': args.start,
               'duration_s': args.duration, 'acoustic_threshold_uncertainty_ms': 20,
               'latency_definition': 'monotonic client input acoustic end to first decoded speech onset; LiveKit without SIP',
               'trials': []}
    for _ in range(args.repeats):
        results['trials'].append(await trial(args, raw, evidence, env))
        results['latency_ms'] = stats(t['latency_ms'] for t in results['trials'] if t.get('latency_ms') is not None)
        (args.output_dir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        if results['trials'][-1]['cleanup_errors'] or results['trials'][-1]['room_leak'] is not False:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--start', type=float, default=8.5)
    parser.add_argument('--duration', type=float, default=2.2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.repeats < 1:
        parser.error('start must be nonnegative; duration and repeats must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
