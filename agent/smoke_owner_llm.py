"""Real LLM tool-choice smoke checks, no phone calls, no tool execution."""
import asyncio
import json
import openai
import gen2b_agent as g

async def main():
    tool = {'type':'function','function':{'name':'ask_owner', 'description':g.Gen2BAssistant.ask_owner._func.__doc__, 'parameters':{'type':'object','properties':{'question':{'type':'string'},'context':{'type':'string'},'options':{'type':'array','items':{'type':'string'},'minItems':2,'maxItems':4}},'required':['question']}}}
    cases = [
        ('delivery_address','Заказать доставку документов сегодня.','Адрес доставки Армен не указал.','Здравствуйте, я личный ассистент Армена Атаяна. Хочу заказать доставку документов.','Да, доставим. На какой адрес доставить?'),
        ('service_substitution','Заказать ремонт с оригинальной деталью.','Замену не разрешено подтверждать без Армена.','Здравствуйте, я личный ассистент Армена Атаяна. Можно заказать ремонт с оригинальной деталью?','Оригинала нет, можем поставить аналог. Согласны?'),
        ('booking_alternative','Забронировать стол на четверых 9 сентября 2026 в 20:00.','Бронь на имя Армен Атаян. Альтернативы только через согласование.','Здравствуйте, я личный ассистент Армена Атаяна. Нужен стол на четверых 9 сентября в 20:00.','На 20:00 занято, есть в 21:00. Забронировать?'),
        ('billiard_choice','Забронировать бильярдный стол завтра в 19:00.','','Здравствуйте. Есть свободный бильярдный стол завтра в 19:00?','Хотите русский бильярд или пул?'),
        ('room_choice','Забронировать пул завтра в 19:00.','','Тогда забронируйте пул на завтра в 19:00.','Какой зал выбрать: обычный или VIP?'),
        ('ambiguous_deposit','Забронировать пять бильярдных столов завтра в 20:00.','Сумму депозита нельзя выдумывать или подтверждать за собеседника.','Какой размер депозита?','Еще раз повторите сто двадцать тысяч рублей это депозит да.'),
        ('opening_no_clock','Уточнить, какие бильярдные столы есть завтра в 20:00, и забронировать.','','','Алло.'),
    ]
    async def run(case):
        name,task,details,previous,utterance=case
        config=g.resolve_call_config(json.dumps({'persona':'armen_personal_assistant','task':task,'task_details':details,'target_name':'сотрудник организации'}))
        gateway = config.get('llm_route') == 'gateway'
        client = openai.AsyncOpenAI(
            base_url=g.GEN2B_BASE if gateway else g.GEN2B_LLM_BASE,
            api_key=g.GEN2B_KEY if gateway else g.GEN2B_LLM_KEY,
            timeout=30,
        )
        try:
            r=await client.chat.completions.create(model=config['llm_model'],messages=[{'role':'system','content':config['system_prompt']},{'role':'assistant','content':previous},{'role':'user','content':utterance}],tools=[tool],tool_choice='auto')
        finally:
            await client.close()
        m=r.choices[0].message
        calls=[{'name':t.function.name,'args':json.loads(t.function.arguments)} for t in m.tool_calls or []]
        print(json.dumps({'case':name,'content':m.content,'calls':calls},ensure_ascii=False),flush=True)
        owner = next((t for t in calls if t['name']=='ask_owner'), None)
        options_ok = True
        if name in {'billiard_choice', 'room_choice'}:
            effective = owner and (owner['args'].get('options') or g.infer_owner_options(owner['args'].get('question', ''), owner['args'].get('context', '')))
            options_ok = bool(effective and len(effective) >= 2)
        if name == 'ambiguous_deposit':
            text=(m.content or '').lower()
            return not calls and '?' in text and 'да, верно' not in text
        if name == 'opening_no_clock':
            text=(m.content or '').lower()
            return not calls and 'сегодня' not in text and '15:47' not in text and '20:00' in text
        return bool(owner) and options_ok
    try:
        results=await asyncio.gather(*(run(c) for c in cases))
        print('TOOL_CHOICE', results, flush=True)
        if not all(results): raise SystemExit(1)
    finally:
        pass

asyncio.run(main())
