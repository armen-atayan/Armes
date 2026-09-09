FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gen2b_agent.py call_control.py .

# Pre-download plugin files (Silero VAD model etc.) at build time
RUN python -m livekit.agents download-files

CMD ["python", "gen2b_agent.py", "start"]
