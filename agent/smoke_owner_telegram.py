"""Manual live smoke test: real Telegram bridge, NO telephone call/action."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import gen2b_agent as g

async def main():
    agent = object.__new__(g.Gen2BAssistant)
    agent._room_name = 'universal-tool-smoke-no-phone-call'
    ctx = SimpleNamespace(session=SimpleNamespace(say=lambda *a, **kw: None))
    result = await g.Gen2BAssistant.ask_owner._func(
        agent, ctx,
        question='Тест универсального уточнения: какое указание вернуть агенту? Нажми «Другое» и напиши, например: «Уточни другой адрес».',
        context='Техническая проверка без телефонного звонка. Реальных заказов или действий не выполняем. Проверяем доставку твоего текста в инструмент.',
    )
    Path('/tmp/universal-owner-smoke-result.json').write_text(result)
    print(result, flush=True)

if __name__ == '__main__':
    asyncio.run(main())
