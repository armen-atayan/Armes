#!/usr/bin/env python3
"""
Place an outbound SIP call with a chosen voice-agent persona.

Usage:
    python call_with_persona.py <persona_key> <phone_number> [--name "Имя"]
        [--pronoun-nom он|она] [--pronoun-acc его|её] [--identity slug]
        [--task "Что сделать"] [--details "Условия и альтернативы"]

Examples:
    python call_with_persona.py angry_grandpa +77077080038 \\
        --name "Армен" --pronoun-nom он --pronoun-acc его --identity armen

    python call_with_persona.py flower_trader +77777033852 \\
        --name "Саша" --pronoun-nom она --pronoun-acc её --identity sasha

Requires LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in the environment
(same as the `lk` CLI), and a SIP_TRUNK_ID either passed via --trunk or read
from the SIP_TRUNK_ID env var.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from livekit import api

PERSONAS_PATH = Path(__file__).parent / "personas.json"


def load_personas() -> dict:
    with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dispatch_metadata(
    *, persona: str, target_name: str, target_pronoun_nom: str,
    target_pronoun_acc: str, target_identity: str, task: str = "",
    task_details: str = "", origin: str = "telegram",
    owner_channel: str = "telegram", result_channel: str = "telegram",
    demo_session_id: str = "",
) -> dict:
    """Build structured job metadata consumed by the voice worker."""
    return {
        "persona": persona,
        "target_name": target_name,
        "target_pronoun_nom": target_pronoun_nom,
        "target_pronoun_acc": target_pronoun_acc,
        "target_identity": target_identity,
        "task": task,
        "task_details": task_details,
        "origin": origin,
        "owner_channel": owner_channel,
        "result_channel": result_channel,
        "demo_session_id": demo_session_id,
    }


def resolve_trunk_id(explicit: str, persona: dict, global_default: str) -> str:
    """Prefer an explicit override, then the persona trunk, then the default."""
    return explicit or str(persona.get("sip_trunk_id", "")) or global_default


def resolve_sip_number(explicit: str, persona: dict) -> str:
    """Prefer an explicit per-call number, then the persona's configured number."""
    return explicit or str(persona.get("sip_number", ""))


async def dispatch_call(
    *,
    persona: str,
    phone_number: str,
    target_name: str = "собеседник",
    target_pronoun_nom: str = "он",
    target_pronoun_acc: str = "его",
    target_identity: str = "callee",
    task: str = "",
    task_details: str = "",
    room_name: str = "",
    origin: str = "telegram",
    owner_channel: str = "telegram",
    result_channel: str = "telegram",
    demo_session_id: str = "",
    trunk_id: str = "",
    sip_number: str = "",
    agent_name: str = "",
    livekit_api=None,
) -> dict:
    """Dispatch one call; callers may inject a LiveKitAPI for tests."""
    personas = load_personas()
    if persona not in personas:
        raise ValueError(f"Unknown persona {persona!r}")
    if persona == "armen_personal_assistant" and not task.strip():
        raise ValueError("The armen_personal_assistant persona requires a task")
    selected = personas[persona]
    trunk_id = resolve_trunk_id(trunk_id, selected, os.environ.get("SIP_TRUNK_ID", ""))
    if not trunk_id:
        raise ValueError("No SIP trunk id configured")
    sip_number = resolve_sip_number(sip_number, selected)
    dial_number = phone_number.removeprefix("+")
    room_name = room_name or f"{persona}-{target_identity}-{int(time.time())}"
    metadata = json.dumps(build_dispatch_metadata(
        persona=persona,
        target_name=target_name,
        target_pronoun_nom=target_pronoun_nom,
        target_pronoun_acc=target_pronoun_acc,
        target_identity=target_identity,
        task=task.strip(),
        task_details=task_details.strip(),
        origin=origin,
        owner_channel=owner_channel,
        result_channel=result_channel,
        demo_session_id=demo_session_id,
    ), ensure_ascii=False)
    owns_client = livekit_api is None
    lkapi = livekit_api or api.LiveKitAPI()
    try:
        await lkapi.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
            room=room_name,
            agent_name=agent_name or os.environ.get("GEN2B_AGENT_NAME", "gen2b-agent"),
            metadata=metadata,
        ))
        participant = await lkapi.sip.create_sip_participant(api.CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id,
            sip_call_to=dial_number,
            sip_number=sip_number,
            room_name=room_name,
            participant_identity=target_identity,
            wait_until_answered=True,
        ))
    finally:
        if owns_client:
            await lkapi.aclose()
    return {
        "room_name": room_name,
        "persona": persona,
        "persona_name": selected["name"],
        "target_name": target_name,
        "phone": phone_number,
        "sip_number": sip_number,
        "sip_call_id": participant.sip_call_id,
        "participant_id": participant.participant_id,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("persona", help="Persona key from personas.json")
    parser.add_argument("phone", help="E.164 phone number to call, e.g. +77077080038")
    parser.add_argument("--name", default="собеседник", help="Target's display name for the prompt")
    parser.add_argument("--pronoun-nom", default="он", help="Target pronoun, nominative (он/она)")
    parser.add_argument("--pronoun-acc", default="его", help="Target pronoun, accusative (его/её)")
    parser.add_argument("--identity", default="callee", help="SIP participant identity slug")
    parser.add_argument("--task", default="", help="Concrete task for a dynamic assistant persona")
    parser.add_argument("--details", default="", help="Task constraints, booking details, and acceptable alternatives")
    parser.add_argument("--trunk", default="", help="SIP outbound trunk ID override")
    parser.add_argument("--sip-number", default="", help="Outbound SIP number / num route parameter override")
    parser.add_argument("--agent-name", default=os.environ.get("GEN2B_AGENT_NAME", "gen2b-agent"), help="Agent name to dispatch (must match worker's agent_name)")
    parser.add_argument("--room", default="", help="Room name (auto-generated if omitted)")
    args = parser.parse_args()

    personas = load_personas()
    if args.persona not in personas:
        print(f"Unknown persona {args.persona!r}. Available: {', '.join(personas)}", file=sys.stderr)
        return 1

    trunk_id = resolve_trunk_id(args.trunk, personas[args.persona], os.environ.get("SIP_TRUNK_ID", ""))
    sip_number = resolve_sip_number(args.sip_number, personas[args.persona])
    if not trunk_id:
        print("No SIP trunk id given (--trunk, persona sip_trunk_id, or SIP_TRUNK_ID env var)", file=sys.stderr)
        return 1

    if args.persona == "armen_personal_assistant" and not args.task.strip():
        print("The armen_personal_assistant persona requires --task", file=sys.stderr)
        return 1

    room_name = args.room or f"{args.persona}-{args.identity}-{int(time.time())}"
    metadata = json.dumps(build_dispatch_metadata(
        persona=args.persona,
        target_name=args.name,
        target_pronoun_nom=args.pronoun_nom,
        target_pronoun_acc=args.pronoun_acc,
        target_identity=args.identity,
        task=args.task.strip(),
        task_details=args.details.strip(),
    ), ensure_ascii=False)

    lkapi = api.LiveKitAPI()
    try:
        # Explicit dispatch carrying the persona/target metadata for the worker.
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=args.agent_name,
                metadata=metadata,
            )
        )

        sip_participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=args.phone,
                sip_number=sip_number,
                room_name=room_name,
                participant_identity=args.identity,
                wait_until_answered=True,
            )
        )
    finally:
        await lkapi.aclose()

    print(json.dumps({
        "room_name": room_name,
        "persona": args.persona,
        "persona_name": personas[args.persona]["name"],
        "target_name": args.name,
        "phone": args.phone,
        "sip_number": sip_number,
        "sip_call_id": sip_participant.sip_call_id,
        "participant_id": sip_participant.participant_id,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
