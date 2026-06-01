#!/usr/bin/env bash
# Launch ChemScape (Python). Reachable at http://130.237.250.75:5013
# Override host/port:  HOST=127.0.0.1 PORT=8077 ./run.sh
set -e
ENV=/home/evehom/Programs/miniconda3/envs/chemscape
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5013}"
cd "$(dirname "$0")/backend"
exec "$ENV/bin/python" -m uvicorn app:app --host "$HOST" --port "$PORT" "$@"
