from __future__ import annotations

import os

import httpx


class Gen2BTranscriber:
    def __init__(self) -> None:
        self.base_url = os.getenv("GEN2B_STT_BASE", "http://100.104.4.1:8093/v1").rstrip("/")
        direct_key = os.getenv("GEN2B_STT_KEY", "")
        gateway_key = os.getenv("GEN2B_KEY", "")
        self.api_key = gateway_key if "ai-kz.gen2b.ai" in self.base_url else direct_key
        self.model = os.getenv("GEN2B_STT_MODEL", "gen2asr")

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        if not audio:
            raise ValueError("empty audio")
        extension = "ogg" if "ogg" in content_type else "webm"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files={"file": (f"voice.{extension}", audio, content_type)},
                data={"model": self.model, "language": "kk_ru"},
            )
            response.raise_for_status()
        text = str(response.json().get("text") or "").strip()
        if not text:
            raise ValueError("empty transcription")
        return text
