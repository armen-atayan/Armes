import os
import subprocess
import sys


def test_worker_defaults_to_current_gen2b_stt_model():
    env = os.environ.copy()
    env.pop("GEN2B_STT_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import gen2b_agent; print(gen2b_agent.GEN2B_STT_MODEL)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "gen2b/stt"
