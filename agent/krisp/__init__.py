"""Krisp VIVA SDK plugin for LiveKit Agents.

Provides noise reduction, voice activity detection, end-of-turn detection,
and interrupt detection backed by the Krisp VIVA SDK.
"""

from krisp.krisp_instance import KrispSDKManager
from krisp.viva_filter import KrispVivaFilterFrameProcessor
from krisp.viva_interrupt_detector import KrispVivaInterruptDetector
from krisp.viva_turn_detector import KrispVivaTurnDetector
from krisp.viva_vad import KrispVivaVAD
from krisp.shared_vad import FanoutVAD
from krisp.turn_rescue import (
    attach_backchannel_gate,
    mark_accumulate_parallel,
    mark_no_rescue,
    patch_false_interruption_min_timeout,
    patch_resume_on_empty_stt_final,
    patch_stale_turn_commit,
    patch_uninterruptible_turn_drop,
    set_keep_all_user_speech,
)

__all__ = [
    "KrispSDKManager",
    "KrispVivaFilterFrameProcessor",
    "KrispVivaInterruptDetector",
    "KrispVivaTurnDetector",
    "KrispVivaVAD",
    "FanoutVAD",
    "attach_backchannel_gate",
    "mark_accumulate_parallel",
    "mark_no_rescue",
    "patch_false_interruption_min_timeout",
    "patch_resume_on_empty_stt_final",
    "patch_stale_turn_commit",
    "patch_uninterruptible_turn_drop",
    "set_keep_all_user_speech",
]
