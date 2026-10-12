#!/usr/bin/env bash
# Resumable download of Charades_v1_480.zip with size verification.
# Charades is non-commercial research data - see DATA_LICENSE.md before use.
set -uo pipefail

URL="https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip"
DEST="${1:-data/raw/Charades_v1_480.zip}"
EXPECTED=16339546533
mkdir -p "$(dirname "$DEST")"

for attempt in $(seq 1 40); do
  have=$([ -f "$DEST" ] && stat -f%z "$DEST" 2>/dev/null || echo 0)
  if [ "$have" -ge "$EXPECTED" ]; then
    echo "complete: $have bytes"; exit 0
  fi
  pct=$(( have * 100 / EXPECTED ))
  echo "[attempt $attempt] have ${have} / ${EXPECTED} bytes (${pct}%) - resuming"
  curl -sSL --fail --retry 3 --retry-delay 5 \
       --speed-time 60 --speed-limit 51200 \
       -C - -o "$DEST" "$URL"
  rc=$?
  echo "[attempt $attempt] curl rc=$rc"
done

have=$(stat -f%z "$DEST" 2>/dev/null || echo 0)
[ "$have" -ge "$EXPECTED" ] && { echo "complete: $have"; exit 0; }
echo "INCOMPLETE: $have / $EXPECTED"; exit 1
