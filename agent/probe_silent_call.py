"""Probe worker-effective STT, LLM and TTS; no telephone call or Telegram send."""
import os,subprocess,asyncio,json,time,io,wave
from pathlib import Path
pid=subprocess.check_output(['systemctl','--user','show','gen2b-agent.service','-p','MainPID','--value'],text=True).strip()
os.environ.update(dict(x.decode().split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x))
import gen2b_agent as g
import httpx,numpy as np
from livekit.agents import llm
from livekit.plugins import openai

async def main():
    cfg=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','target_name':'сотрудник ресторана','task':'Забронировать стол на четверых 9 сентября 2026 в 20:00.','task_details':'На имя Армен Атаян. Другие варианты согласовать в Telegram.'}))
    agent=g.Gen2BAssistant(room_name='diagnostic-no-call',participant_identity='test',on_outcome_saved=lambda:None,system_prompt=cfg['system_prompt'],recording_path_getter=lambda:None)
    async def stt():
        p=Path('../call-recordings/armen_personal_assistant-armen-callback-retest-1788881694.ogg')
        print('SOURCE',p.resolve(),p.stat().st_size,flush=True)
        async with httpx.AsyncClient(timeout=90) as c:
            r=await c.post(g.GEN2B_STT_BASE+'/audio/transcriptions',headers={'Authorization':'Bearer '+g.GEN2B_STT_KEY},data={'model':g.GEN2B_STT_MODEL,'language':'kk_ru'},files={'file':(p.name,p.read_bytes(),'audio/ogg')})
            print('RECORDING_STT',r.status_code,r.text[:6000],flush=True)
    async def model():
        selected=os.environ.get('PROBE_LLM_MODEL',cfg['llm_model'])
        print('LLM_MODEL',selected,flush=True)
        gateway=os.environ.get('PROBE_LLM_ROUTE',cfg.get('llm_route'))=='gateway'
        m=openai.LLM(model=selected,base_url=g.GEN2B_BASE if gateway else g.GEN2B_LLM_BASE,api_key=g.GEN2B_KEY if gateway else g.GEN2B_LLM_KEY,reasoning_effort=g.GEN2B_LLM_REASONING_EFFORT)
        try:
            for attempt in range(2):
                ctx=llm.ChatContext();ctx.add_message(role='system',content=cfg['system_prompt']);ctx.add_message(role='user',content='Алло.')
                chunks=[];started=time.monotonic()
                async with m.chat(chat_ctx=ctx,tools=agent.tools) as stream:
                    async for chunk in stream: chunks.append(chunk.model_dump())
                print('LLM_GREETING',attempt,round(time.monotonic()-started,3),json.dumps({'text':''.join((x.get('delta') or {}).get('content') or '' for x in chunks),'chunks':len(chunks),'usage':chunks[-1].get('usage') if chunks else None},ensure_ascii=False),flush=True)
        finally:await m.aclose()
    async def tts():
        t=openai.TTS(model=g.GEN2B_TTS_MODEL,voice='clone:armen2',base_url=g.GEN2B_DIRECT_TTS_BASE,api_key=g._GEN2B_DIRECT_TTS_KEY,response_format='pcm')
        try:
            frames=[];rates=set();started=time.monotonic()
            async with t.synthesize('Здравствуйте, я личный ассистент Армена Атаяна.') as s:
                async for e in s: frames.append(bytes(e.frame.data));rates.add(e.frame.sample_rate)
            raw=b''.join(frames);a=np.frombuffer(raw,dtype='<i2').astype(float)/32768
            print('TTS_SDK',{'seconds':round(time.monotonic()-started,3),'frames':len(frames),'bytes':len(raw),'sample_rates':list(rates),'rms_db':round(float(20*np.log10(np.sqrt(np.mean(a*a))+1e-12)),2),'peak_db':round(float(20*np.log10(np.max(abs(a))+1e-12)),2)},flush=True)
            with wave.open('/tmp/armen2-current-tts.wav','wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(next(iter(rates)));w.writeframes(raw)
        finally:await t.aclose()
    if os.environ.get('PROBE_LLM_MODEL'):await model()
    else:await asyncio.gather(stt(),model(),tts())
asyncio.run(main())
