"""Offline route and native streaming request contracts (no provider calls)."""
import ast
import importlib
import inspect
import json
from unittest.mock import AsyncMock, Mock

import pytest
from livekit.agents import function_tool
from livekit.agents.llm import ChatContext
from livekit.plugins import anthropic
from livekit.plugins.anthropic import llm as native
import anthropic as sdk

import gen2b_agent as g


@pytest.fixture
def factory():
    return importlib.import_module('llm_factory')


@pytest.fixture
def legacy():
    return dict(gateway_base='https://gateway.invalid/v1', gateway_key='gateway-test',
                default_base='https://legacy.invalid/v1', default_key='legacy-test',
                reasoning_effort='none')


def test_missing_persona_overrides_use_gemini_gateway(monkeypatch):
    personas = g.load_personas()
    for persona in personas.values():
        persona.pop('llm_model', None)
        persona.pop('llm_route', None)
    monkeypatch.setattr(g, 'load_personas', lambda: personas)
    # The resolver legitimately honors the import-time environment override.
    # Pin it here so this test isolates missing persona fields, not shell state.
    assert g.DEFAULT_LLM_MODEL == 'gemini-2.5-flash'
    monkeypatch.setattr(g, 'GEN2B_LLM_MODEL', g.DEFAULT_LLM_MODEL)
    for name in personas:
        cfg = g.resolve_call_config(json.dumps({'persona': name}))
        assert (cfg['llm_model'], cfg['llm_route']) == ('gemini-2.5-flash', 'gateway')


def test_worker_uses_shared_factory():
    tree = ast.parse(inspect.getsource(g.entrypoint))
    session = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'AgentSession')
    llm = next(k.value for k in session.keywords if k.arg == 'llm')
    assert isinstance(llm, ast.Call) and isinstance(llm.func, ast.Name)
    assert llm.func.id == 'create_llm'


@pytest.mark.parametrize('key', [None, '', '   '])
def test_anthropic_requires_dedicated_key(factory, legacy, monkeypatch, key):
    if key is None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    else:
        monkeypatch.setenv('ANTHROPIC_API_KEY', key)
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        factory.create_llm({'llm_route': 'anthropic', 'llm_model': 'claude-haiku-5-5'}, **legacy)


@pytest.mark.parametrize('route', ['gateway', 'default'])
def test_legacy_routes_preserved(factory, legacy, monkeypatch, route):
    constructor = Mock()
    monkeypatch.setattr(factory.openai, 'LLM', constructor)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    result = factory.create_llm({'llm_route': route, 'llm_model': 'legacy-model'}, **legacy)
    prefix = 'gateway' if route == 'gateway' else 'default'
    constructor.assert_called_once_with(model='legacy-model',
        base_url=legacy[prefix + '_base'], api_key=legacy[prefix + '_key'], reasoning_effort='none')
    assert result is constructor.return_value


@pytest.mark.asyncio
@pytest.mark.parametrize('config', [{'llm_route': 'anthropic', 'llm_model': 'claude-haiku-5-5'}])
@pytest.mark.parametrize('trailing_assistant', [False, True])
async def test_native_streaming_with_tools(factory, legacy, monkeypatch, config, trailing_assistant):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-offline-test')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://wrong.invalid')
    model = factory.create_llm(config, **legacy)
    assert isinstance(model, anthropic.LLM)
    assert model.model == 'claude-haiku-5-5'
    assert str(model._client.base_url).rstrip('/') == 'https://api.anthropic.com'
    assert model._client.api_key == 'anthropic-offline-test'
    create = AsyncMock(return_value=object())
    monkeypatch.setattr(model._client.messages, 'create', create)
    stream_constructor = Mock()
    monkeypatch.setattr(native, 'LLMStream', stream_constructor)

    @function_tool
    async def ask_owner(question: str) -> str:
        """Ask the owner for a decision."""
        return question

    ctx = ChatContext()
    ctx.add_message(role='user', content='Ask the owner to confirm')
    if trailing_assistant:
        ctx.add_message(role='assistant', content='Let me check.')
    original_items = list(ctx.items)
    try:
        model.chat(chat_ctx=ctx, tools=[ask_owner])
        await stream_constructor.call_args.kwargs['create_anthropic_stream']()
        request = create.call_args.kwargs
        assert request['messages'][-1]['role'] == 'user'
        assert ctx.items == original_items
        # Bind against the real SDK API, not a permissive mock signature.
        inspect.signature(sdk.resources.messages.AsyncMessages.create).bind(None, **request)
        assert request['model'] == 'claude-haiku-5-5'
        assert request['stream'] is True
        assert request['thinking'] == {'type': 'disabled'}
        assert 'output_config' not in request
        assert request['tool_choice'] == {'type': 'auto'}
        assert request['tools'][0]['name'] == 'ask_owner'
        assert request['tools'][0]['input_schema']['required'] == ['question']
        assert 'temperature' not in request
        assert 'reasoning_effort' not in request
    finally:
        await model.aclose()
