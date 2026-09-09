"""LiveKit room controls and durable call outcomes for outbound SIP calls."""

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from livekit import api
from livekit.protocol import room


async def hang_up_sip_participant(
    room_name: str,
    participant_identity: str,
    api_factory: Callable[[], Any] = api.LiveKitAPI,
) -> None:
    """End a SIP call by removing its participant from the LiveKit room."""
    lkapi = api_factory()
    try:
        await lkapi.room.remove_participant(
            room.RoomParticipantIdentity(
                room=room_name,
                identity=participant_identity,
            )
        )
    finally:
        await lkapi.aclose()


async def start_call_recording(
    room_name: str,
    recordings_dir: Path,
    api_factory: Callable[[], Any] = api.LiveKitAPI,
) -> str | None:
    """Start an audio-only room-composite egress that records the call to a
    local WAV/OGG file mounted into the livekit-egress container at /out.

    Returns the egress_id (needed to stop it later) or None on failure —
    recording is best-effort and must never block/crash the call itself.
    """
    recordings_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{room_name.replace('/', '_')}.ogg"
    # /out is the mount point inside the livekit-egress container; the host
    # path is recordings_dir (same volume, different mountpoints).
    filepath = f"/out/{filename}"

    lkapi = api_factory()
    try:
        info = await lkapi.egress.start_room_composite_egress(
            api.RoomCompositeEgressRequest(
                room_name=room_name,
                audio_only=True,
                file_outputs=[
                    api.EncodedFileOutput(
                        file_type=api.EncodedFileType.OGG,
                        filepath=filepath,
                    )
                ],
            )
        )
        return info.egress_id
    except Exception:
        return None
    finally:
        await lkapi.aclose()


async def stop_call_recording(
    egress_id: str,
    api_factory: Callable[[], Any] = api.LiveKitAPI,
) -> None:
    """Stop an in-progress egress. Best-effort — swallow errors since this
    typically runs during call teardown."""
    lkapi = api_factory()
    try:
        await lkapi.egress.stop_egress(api.StopEgressRequest(egress_id=egress_id))
    except Exception:
        pass
    finally:
        await lkapi.aclose()


def write_call_outcome(
    *,
    results_dir: Path,
    room_name: str,
    outcome: str,
    summary: str,
    agreed_price_kzt: int | None,
    volume_m3: int | None,
    next_step: str,
    agreed_price_gbp: float | None = None,
    recording_path: str | None = None,
) -> Path:
    """Atomically persist a pending outcome for the Telegram delivery worker."""
    pending_dir = results_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{room_name.replace('/', '_')}.json"
    destination = pending_dir / filename
    temporary = pending_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
    payload = {
        "room_name": room_name,
        "outcome": outcome,
        "summary": summary,
        "agreed_price_kzt": agreed_price_kzt,
        "agreed_price_gbp": agreed_price_gbp,
        "volume_m3": volume_m3,
        "next_step": next_step,
        "recording_path": recording_path,
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
    return destination

