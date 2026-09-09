"""Live speech-style probe; never executes tools or places a call."""
import os, subprocess, json, asyncio, re
from pathlib import Path
pid=subprocess.check_output(['systemctl','--user','show','gen2b-agent.service','-p','MainPID','--value'],text=True).strip()
os.environ.update(dict(x.decode().split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x))
import gen2b_agent as g
from livekit.agents import llm
from livekit.plugins import openai

async def main():
    cfg=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','target_name':'сотрудник ресторана','task':'Позвони в ресторан и забронируй стол на четверых на 9 сентября 2026 года в 20:00 на имя Армен Атаян.','task_details':'Ролевая проверка ресторана. Другие варианты согласуй через ask_owner. Без лишних денежных вопросов.'}))
    root=Path(__file__).parent/'artifacts';root.mkdir(exist_ok=True)
    (root/'armen-assistant-full-prompt.txt').write_text('ТЕКУЩИЙ SYSTEM PROMPT, собранный для тестового ресторанного поручения.\nВключает: персона + поручение + global_tone.md + OWNER_CLARIFICATION_POLICY.\nМодель: '+cfg['llm_model']+'\n\n'+cfg['system_prompt']+'\n\nGREETING INSTRUCTIONS (отдельный слой; при ожидании первой реплики не запускается проактивно):\n'+cfg['greeting_instructions'])
    agent=g.Gen2BAssistant(room_name='style-probe-no-call',participant_identity='test',on_outcome_saved=lambda:None,system_prompt=cfg['system_prompt'],recording_path_getter=lambda:None)
    selected=os.environ.get('STYLE_PROBE_MODEL',cfg['llm_model'])
    gateway=os.environ.get('STYLE_PROBE_ROUTE',cfg.get('llm_route'))=='gateway'
    print('PROBE_MODEL',selected,'ROUTE', 'gateway' if gateway else 'default',flush=True)
    m=openai.LLM(model=selected,base_url=g.GEN2B_BASE if gateway else g.GEN2B_LLM_BASE,api_key=g.GEN2B_KEY if gateway else g.GEN2B_LLM_KEY,reasoning_effort=g.GEN2B_LLM_REASONING_EFFORT)
    opener='Здравствуйте, я личный ассистент Армена Атаяна. Есть стол на четверых девятого сентября в восемь вечера?'
    cases={'greeting':[('user','Алло.')], 'confirmation':[('user','Алло.'),('assistant',opener),('user','Да, свободно, на имя Армена Атаяна, четверо гостей.')], 'correction':[('user','Алло.'),('assistant',opener),('user','Нет, в восемь вечера всё занято.')]}
    results=[]
    try:
        for name,history in list(cases.items()) + [('greeting_repeat_'+str(i),cases['greeting']) for i in range(3)]:
            ctx=llm.ChatContext();ctx.add_message(role='system',content=cfg['system_prompt'])
            for role,text in history:ctx.add_message(role=role,content=text)
            chunks=[]
            async with m.chat(chat_ctx=ctx,tools=agent.tools) as stream:
                async for x in stream:chunks.append(x.model_dump())
            text=''.join((x.get('delta') or {}).get('content') or '' for x in chunks)
            tool_calls=[t for x in chunks for t in (x.get('delta') or {}).get('tool_calls',[]) or []]
            # A tool-only turn is not a spoken utterance; tools are never executed here.
            passed=(bool(text) or (name=='correction' and bool(tool_calls))) and not re.search(r'\bпонял\b|\b2026\b|две тысячи двадцать шест|подтвердите',text,re.I)
            if name=='confirmation':passed=passed and bool(re.search(r'можете[\s,]+пожалуйста[\s,]+подтвердить',text,re.I))
            row={'case':name,'text':text,'pass':passed,'tool_calls':[t for x in chunks for t in (x.get('delta') or {}).get('tool_calls',[]) or []]};results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:await m.aclose()
    (root/'armen-style-live-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    assert all(r['pass'] for r in results), 'Live style constraint failed'

asyncio.run(main())
