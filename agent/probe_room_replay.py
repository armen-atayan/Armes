"""Inject original callee hello into the live worker without any SIP call."""
import os,asyncio,subprocess,json,time,wave
from pathlib import Path
pid=subprocess.check_output(['systemctl','--user','show','gen2b-agent.service','-p','MainPID','--value'],text=True).strip()
os.environ.update(dict(x.decode().split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x))
from livekit import rtc,api
import numpy as np

async def main():
    name='diagnostic-armen-silence-'+str(int(time.time()))
    client=api.LiveKitAPI(url='http://localhost:7880')
    room=rtc.Room();tasks=[];chunks=[];transcripts=[];out_ready=asyncio.Event()
    async def capture(track):
        async for e in rtc.AudioStream(track,sample_rate=24000,num_channels=1):chunks.append(bytes(e.frame.data))
    @room.on('track_subscribed')
    def on_track(track,pub,participant):
        if track.kind==rtc.TrackKind.KIND_AUDIO:
            print('OUTPUT_TRACK',participant.identity,flush=True)
            tasks.append(asyncio.create_task(capture(track)));out_ready.set()
    @room.on('transcription_received')
    def on_text(segments,participant,pub):
        for seg in segments:
            if seg.final:
                transcripts.append({'identity':participant.identity if participant else '', 'text':seg.text})
                print('TRANSCRIPT',transcripts[-1],flush=True)
    token=(api.AccessToken().with_identity('diagnostic-callee').with_name('Диагностический аудиотест без звонка').with_grants(api.VideoGrants(room_join=True,room=name)).with_attributes({'sip.callStatus':'active'}).to_jwt())
    try:
        await room.connect('ws://localhost:7880',token)
        source=rtc.AudioSource(8000,1)
        track=rtc.LocalAudioTrack.create_audio_track('original-callee-hello',source)
        await room.local_participant.publish_track(track,rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        meta={'persona':'armen_personal_assistant','target_name':'диагностический стенд — без телефонного звонка','task':'Технический тест: после «Алло» представься личным ассистентом Армена Атаяна и спроси, хорошо ли тебя слышно. Никаких заказов и звонков не выполняй.'}
        await client.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(agent_name='gen2b-agent',room=name,metadata=json.dumps(meta,ensure_ascii=False)))
        print('ROOM',name,flush=True)
        await asyncio.wait_for(out_ready.wait(),20)
        hello=subprocess.check_output(['ffmpeg','-v','error','-ss','8.5','-t','2.2','-i','../call-recordings/armen_personal_assistant-armen-callback-retest-1788881694.ogg','-f','s16le','-ac','1','-ar','8000','-'])
        raw=bytes(8000*2)+hello+bytes(8000*2*8)+hello+bytes(8000*2*9)
        started=time.monotonic()
        for pos in range(0,len(raw),320):
            block=raw[pos:pos+320].ljust(320,b'\0')
            await source.capture_frame(rtc.AudioFrame(block,8000,1,160))
        await source.wait_for_playout()
        print('INPUT_COMPLETE',round(time.monotonic()-started,2),flush=True)
    finally:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        await room.disconnect()
        await client.room.delete_room(api.DeleteRoomRequest(room=name))
        await client.aclose()
    raw=b''.join(chunks)
    p=Path('/tmp/'+name+'-agent.wav')
    with wave.open(str(p),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
    a=np.frombuffer(raw,dtype='<i2').astype(float)/32768
    result={'room':name,'audio':str(p),'bytes':len(raw),'seconds':len(a)/24000,'rms_db':round(float(20*np.log10(np.sqrt(np.mean(a*a))+1e-12)),2) if len(a) else None,'peak_db':round(float(20*np.log10(np.max(abs(a))+1e-12)),2) if len(a) else None,'transcripts':transcripts}
    Path('/tmp/armen-room-replay-result.json').write_text(json.dumps(result,ensure_ascii=False))
    print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
asyncio.run(main())
