#!/bin/bash
# Make a booted pod ready to run this repo, then copy run inputs onto it:
#
#   bash cloud/stage_pod.sh POD_ID runs/p2_f/state.json runs/p2_f/replay/ runs/data/mate1_100k_clean/ ...
#
# Pushes a git bundle of main (the pod never holds GitHub credentials), the pod
# scripts, runs cloud/setup_pod.sh, then copies each repo-relative PATH to the same
# path under /root/chess (a trailing / copies a directory's contents). Everything on
# the pod is reproducible from this box; nothing is fetched from anywhere else.
set -euo pipefail
POD=$1
shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
RP=(.venv/bin/python scripts/runpod_pod.py)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

git bundle create "$STAGE/chess.bundle" main 2> /dev/null
"${RP[@]}" ssh "$POD" -- mkdir -p /root/taz
"${RP[@]}" push "$POD" "$STAGE/chess.bundle" /root/taz/chess.bundle
for f in setup_pod.sh pod_job.sh; do "${RP[@]}" push "$POD" "cloud/$f" "/root/taz/$f"; done
"${RP[@]}" ssh "$POD" -- bash /root/taz/setup_pod.sh /root/taz/chess.bundle 2>&1 | grep -E '^(code|stack|stockfish|cpu):'

t0=$(date +%s)
for path in "$@"; do
    "${RP[@]}" ssh "$POD" -- mkdir -p "/root/chess/$(dirname "$path")"
    "${RP[@]}" push "$POD" "$path" "/root/chess/$path"
done
echo "inputs: $# path(s) in $(($(date +%s) - t0)) s"
