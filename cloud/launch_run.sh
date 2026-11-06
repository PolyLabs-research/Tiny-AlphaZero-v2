#!/bin/bash
# One remote run from nothing to training, unattended:
#
#   cloud/launch_run.sh TAG HOURS RUN "JUDGED" FINAL START_REPS START_STARTPOS JOB INPUT...
#
# Launches a pod (`runpod_pod.py launch`, retried while none is offered), waits for SSH, stages the code and INPUTs
# (repo-relative paths, a directory with a trailing /), checks every input's sha256 on the pod, starts JOB under
# pod_job.sh, then writes ~/.config/taz-cloud/campaigns/RUN.env and enables taz-campaign@RUN.service, which pulls and
# measures from there. Log: runs/RUN_launch.console.log.
#
# A pod this script launched ends up training or terminated, never left to its deadline: p2_r's first cont pod came up
# with no runtime and sat billing until it was noticed. MAX_PODS bounds the replacements.
#
# Environment: WAIT_FOR=PATH waits for a file before launching (a second stage waits for the first's .remote_done),
# PULL_WAIT_S (14400), RETRY_S (300, between refused launches), SSH_WINDOW_S (1200), MAX_PODS (3), TRIES (288).
set -u
TAG=$1; HOURS=$2; RUN=$3; JUDGED=$4; FINAL=$5; START_REPS=$6; START_STARTPOS=$7; JOB=$8; shift 8
REPO=${LAUNCH_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 1
read -r -a RP <<< "${LAUNCH_RP:-.venv/bin/python scripts/runpod_pod.py}"
read -r -a STAGE <<< "${LAUNCH_STAGE:-bash cloud/stage_pod.sh}"
read -r -a SYSTEMCTL <<< "${LAUNCH_SYSTEMCTL:-systemctl --user}"
CAMPAIGNS=${TAZ_CAMPAIGNS:-$HOME/.config/taz-cloud/campaigns}
RETRY_S=${RETRY_S:-300}; SSH_WINDOW_S=${SSH_WINDOW_S:-1200}; MAX_PODS=${MAX_PODS:-3}; TRIES=${TRIES:-288}
PULL_WAIT_S=${PULL_WAIT_S:-14400}
LOG=runs/${RUN}_launch.console.log
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

log() { echo "$(date -u +%FT%T+00:00) $*" >> "$LOG"; }

launch() {  # sets POD, or returns 1 after TRIES refusals
    local out
    POD=""
    for try in $(seq 1 "$TRIES"); do
        out=$("${RP[@]}" launch --gpu "RTX 4090" --cloud SECURE --hours "$HOURS" --tag "$TAG" --vcpu 24 --ram 32 --disk 60 --yes 2>&1)
        POD=$(echo "$out" | grep -oP '^launched \K\S+')
        if [ -n "$POD" ]; then
            log "launched: $(echo "$out" | grep -E '^(plan|launched)' | tr '\n' ' ')"
            return 0
        fi
        log "launch refused ($try of $TRIES): $(echo "$out" | grep -m1 -oE 'error: .{0,90}')"
        sleep "$RETRY_S"
    done
    return 1
}

wait_ssh() {  # an API error is not evidence: retry inside the window, which is what declares a pod dead
    local end=$(($(date +%s) + SSH_WINDOW_S)) left
    while left=$((end - $(date +%s))); [ "$left" -gt 0 ]; do
        "${RP[@]}" wait-ssh "$POD" --timeout "$left" >> "$LOG" 2>&1 && return 0
        sleep $((left < 20 ? left : 20))
    done
    return 1
}

stage() {
    local path a ok
    ok=0
    for a in 1 2 3; do "${STAGE[@]}" "$POD" >> "$LOG" 2>&1 && { ok=1; break; }; sleep 20; done
    [ $ok = 1 ] || { log "setup failed on $POD"; return 1; }
    for path in "${INPUTS[@]}"; do
        ok=0
        for a in 1 2 3 4; do
            "${RP[@]}" ssh "$POD" -- mkdir -p "/root/chess/$(dirname "$path")" >> "$LOG" 2>&1 \
                && "${RP[@]}" push "$POD" "$path" "/root/chess/$path" >> "$LOG" 2>&1 && { ok=1; break; }
            sleep $((2 ** a))
        done
        [ $ok = 1 ] || { log "copy failed on $POD: $path"; return 1; }
    done
    "${RP[@]}" push "$POD" "$WORK/inputs.sha256" /root/taz/inputs.sha256 >> "$LOG" 2>&1 || return 1
    "${RP[@]}" ssh "$POD" -- 'cd /root/chess && sha256sum -c --quiet /root/taz/inputs.sha256 && echo ALL_MATCH' 2>&1 | grep -q ALL_MATCH \
        || { log "sha256 mismatch on $POD"; return 1; }
    log "inputs verified on $POD ($(wc -l < "$WORK/inputs.sha256") files)"
}

start_job() {
    log "gpu: $("${RP[@]}" ssh "$POD" -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tail -1)"
    "${RP[@]}" ssh "$POD" -- "nohup setsid bash /root/taz/pod_job.sh /root/taz/work $PULL_WAIT_S -- bash -c \"cd /root/chess && $JOB\" > /dev/null 2>&1 < /dev/null &" >> "$LOG" 2>&1
}

enable_driver() {
    mkdir -p "$CAMPAIGNS"
    cat > "$CAMPAIGNS/$RUN.env" << EOF
POD=$POD
JUDGED="$JUDGED"
FINAL=$FINAL
MEASURE_AFTER_PULL=1
START_REPS=$START_REPS
START_STARTPOS=$START_STARTPOS
EOF
    "${SYSTEMCTL[@]}" enable --now "taz-campaign@$RUN.service" >> "$LOG" 2>&1
}

INPUTS=("$@")
if [ -n "${WAIT_FOR:-}" ]; then
    log "waiting for $WAIT_FOR"
    until [ -e "$WAIT_FOR" ]; do sleep 60; done
fi
find "${INPUTS[@]}" -type f | sort | xargs sha256sum > "$WORK/inputs.sha256" || { log "cannot checksum the inputs"; exit 1; }

for pod_try in $(seq 1 "$MAX_PODS"); do
    launch || { log "FAILED: no pod offered in $TRIES tries"; exit 1; }
    if wait_ssh && stage && start_job; then
        enable_driver
        log "training: pod $POD, driver taz-campaign@$RUN"
        exit 0
    fi
    log "pod $POD is unusable (try $pod_try of $MAX_PODS); terminating it"
    "${RP[@]}" terminate "$POD" >> "$LOG" 2>&1 || log "FAILED to terminate $POD: its deadline and the sweeper remain"
done
log "FAILED: $MAX_PODS pods unusable"
exit 1
