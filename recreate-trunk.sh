#!/bin/bash
# Recreate the Telnyx outbound SIP trunk if lost (e.g. after a Redis wipe).
# Safe to re-run — creates a new trunk each time (old ones just become orphaned).
set -e
cd "$(dirname "$0")"
export $(grep -v '^#' .env | xargs)
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY
export LIVEKIT_API_SECRET
lk sip outbound create outbound-trunk.json
echo "Now update SIP_TRUNK_ID in your call scripts / dispatch rules with the new ID above."
