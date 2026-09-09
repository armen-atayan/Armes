from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DEMO_DATA_DIR", "/home/ubuntu/livekit-stack/demo-data")))
    callback_dir: Path = field(default_factory=lambda: Path(os.getenv("LIVE_CALLBACK_DIR", "/home/ubuntu/livekit-stack/live-callbacks")))
    recordings_dir: Path = field(default_factory=lambda: Path(os.getenv("CALL_RECORDINGS_DIR", "/home/ubuntu/livekit-stack/call-recordings")))
    frontend_dist: Path = field(default_factory=lambda: Path(os.getenv("DEMO_FRONTEND_DIST", "/home/ubuntu/livekit-stack/demo-ui/dist")))
    demo_token: str | None = field(default_factory=lambda: os.getenv("DEMO_TOKEN") or None)
    websocket_poll_interval: float = field(default_factory=lambda: float(os.getenv("DEMO_WS_POLL_INTERVAL", "0.15")))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.callback_dir = Path(self.callback_dir)
        self.recordings_dir = Path(self.recordings_dir)
        self.frontend_dist = Path(self.frontend_dist)
