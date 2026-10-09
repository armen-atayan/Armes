"""Observability only: native provider events and LiveKit 1.6 per-turn metrics."""
import json
import logging
import math

logger = logging.getLogger('gen2b-agent')
PROVIDER_FIELDS = ('duration', 'ttft', 'ttfb', 'audio_duration', 'cancelled')
TURN_FIELDS = ('transcription_delay', 'end_of_turn_delay', 'on_user_turn_completed_delay',
               'llm_node_ttft', 'tts_node_ttfb', 'playback_latency', 'e2e_latency')


def attach_pipeline_diagnostics(session, stt, llm, tts, room_name):
    """Attach synchronous observation callbacks; never modify streams or turns."""
    def provider(metric):
        values = {key: getattr(metric, key) for key in PROVIDER_FIELDS
                  if isinstance(getattr(metric, key, None), (int, float, bool))
                  and math.isfinite(getattr(metric, key))}
        logger.info('PIPELINE_PROVIDER_METRIC room=%s module=%s metrics=%s',
                    room_name, getattr(metric, 'type', 'unknown'), json.dumps(values))

    def turn(event):
        item = getattr(event, 'item', None)
        metrics = getattr(item, 'metrics', {}) or {}
        if not isinstance(metrics, dict):
            return
        values = {key: metrics[key] for key in TURN_FIELDS
                  if isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])}
        if values:
            logger.info('PIPELINE_TURN_METRIC room=%s role=%s metrics=%s',
                        room_name, getattr(item, 'role', 'unknown'), json.dumps(values))

    for model in (stt, llm, tts):
        if model is not None:
            model.on('metrics_collected', provider)
    session.on('conversation_item_added', turn)
