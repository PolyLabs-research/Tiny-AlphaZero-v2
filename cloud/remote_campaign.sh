#!/bin/bash
# Option B (PREREG-p2_g): a run trains on a pod; this box pulls it and measures the
# judged checkpoints on its own instrument. Started detached, so it outlives the
# session that started it:
#
#   setsid nohup bash cloud/remote_campaign.sh POD_ID RUN "10 20 30 40 50 60" 60 \
#       > /dev/null 2>&1 < /dev/null &
#
# Every PULL_EVERY seconds: pull runs/RUN (without the ring) and the pod's console.
# A judged checkpoint N is measured once state.json has committed past it, so the
# file is long finished: the ladder pass (all four rungs, suites, ECE, book entropy)
# and three skill-0 replays from the frozen book, under one gpu hold; at the final
# checkpoint also three replays from the initial position (the distribution check).
# When the pod's job marks DONE: pull again with the ring, check state.json says the
# final iteration, write PULLED on the pod (its job then deletes it), finish the
# measurements, compute the verdicts. The pod's deadline and the sweeper still cap
# everything if this box is down. Log: runs/RUN_campaign.console.log.
set -u
POD=$1
RUN=$2
JUDGED=$3
FINAL=$4
START_REPS=${START_REPS:-runs/p2_f/reps_iter0020_r*.jsonl}
START_STARTPOS=${START_STARTPOS:-runs/p2_f/startpos/reps_iter0020_r*.jsonl}
PULL_EVERY=${PULL_EVERY:-300}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
PY=.venv/bin/python
RP=("$PY" scripts/runpod_pod.py)
GPU=$HOME/.local/bin/gpu
LOG=runs/${RUN}_campaign.console.log
mkdir -p "runs/$RUN"

log() { echo "$(date -Is) $*" >> "$LOG"; }
pod_alive() { "$PY" -c "import sys; from tinyalphazero.cloud import RunPod; sys.exit(0 if RunPod().get_pod('$POD') else 1)" 2> /dev/null; }
committed() { "$PY" -c "import json; print(json.load(open('runs/$RUN/state.json'))['iteration'])" 2> /dev/null || echo -1; }
pull_run() {
    # seed_* are the donor run's own checkpoints, copied in by --seed-pool; this box has them.
    "${RP[@]}" pull "$POD" "/root/chess/runs/$RUN/" "runs/$RUN/" --exclude='seed_*' "$@" >> "$LOG" 2>&1
    "${RP[@]}" pull "$POD" /root/taz/work/job.log "runs/$RUN.console.log" >> "$LOG" 2>&1
}

measure() {
    local n=$1 nn
    nn=$(printf %04d "$n")
    [ -e "runs/$RUN/.measured_$nn" ] && return 0
    [ -f "runs/$RUN/ckpt_iter$nn.pt" ] || return 1
    for _ in $(seq 1 120); do  # gpu hold refuses while ComfyUI has queued jobs
        q=$("$GPU" status 2> /dev/null | grep -oP 'queue \K[0-9]+' | head -1)
        [ "${q:-0}" = 0 ] && [ "$("$GPU" lease 2> /dev/null)" = "{}" ] && break
        sleep 60
    done
    log "measure $nn: start"
    local startpos=""
    [ "$n" = "$FINAL" ] && startpos="$PY scripts/eval_reps.py run --ckpt runs/$RUN/ckpt_iter$nn.pt --anchor stockfish:0 --reps 3 --openings tests/suites/openings_startpos.epd --out-dir runs/$RUN/startpos"
    if ! "$GPU" hold --by tinyalphazero -- env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTHONUNBUFFERED=1 bash -c "
        set -e
        mkdir -p runs/$RUN/startpos
        $PY scripts/eval_ladder.py --ckpt runs/$RUN/ckpt_iter$nn.pt --config runs/$RUN/config.yaml \
            --anchors random,greedy,phase1,stockfish:0 --anchor-ckpt runs/phase1_anchor.pt \
            --iteration $n --role protocol --out runs/$RUN/ladder.jsonl
        $PY scripts/eval_reps.py run --ckpt runs/$RUN/ckpt_iter$nn.pt --anchor stockfish:0 --reps 3
        $startpos
    " >> "$LOG" 2>&1; then
        log "measure $nn: FAILED (see above); will retry"
        return 1
    fi
    # shellcheck disable=SC2086  # the start globs expand here, on purpose
    "$PY" scripts/eval_reps.py verdict --a start $START_REPS --b "${RUN}_iter$nn" runs/$RUN/reps_iter${nn}_r*.jsonl \
        --out "runs/$RUN/p2_verdict_reps_iter$nn.json" >> "$LOG" 2>&1
    if [ "$n" = "$FINAL" ]; then
        cp "runs/$RUN/p2_verdict_reps_iter$nn.json" "runs/$RUN/p2_verdict_reps.json"
        # shellcheck disable=SC2086
        "$PY" scripts/eval_reps.py verdict --a start_startpos $START_STARTPOS --b "${RUN}_startpos" \
            runs/$RUN/startpos/reps_iter${nn}_r*.jsonl --out "runs/$RUN/startpos/p2_verdict_reps.json" >> "$LOG" 2>&1
    fi
    touch "runs/$RUN/.measured_$nn"
    log "measure $nn: done"
}

finish_pod() {  # the job is done: everything back, then release the pod
    pull_run
    pull_run  # a second pass picks up anything the first saw mid-write
    if [ "$(committed)" = "$FINAL" ] && [ -f "runs/$RUN/replay/meta.json" ]; then
        "${RP[@]}" ssh "$POD" -- touch /root/taz/work/PULLED >> "$LOG" 2>&1
        touch "runs/$RUN/.remote_done"
        log "final pull complete (iteration $FINAL, ring included); PULLED written, the pod deletes itself"
    else
        log "final pull INCOMPLETE: state.json says $(committed), ring meta $([ -f runs/$RUN/replay/meta.json ] && echo present || echo missing); retrying next round"
    fi
}

log "armed: pod $POD run $RUN judged [$JUDGED] final $FINAL"
while :; do
    if [ ! -e "runs/$RUN/.remote_done" ] && pod_alive; then
        pull_run --exclude=replay/ --exclude='*.tmp'
        if "${RP[@]}" ssh "$POD" -- test -e /root/taz/work/DONE 2> /dev/null; then
            finish_pod
        fi
    elif [ ! -e "runs/$RUN/.remote_done" ]; then
        log "pod $POD is gone before a complete final pull (deadline, sweeper, or failure); measuring what is here"
        touch "runs/$RUN/.pod_gone"
    fi

    c=$(committed)
    for n in $JUDGED; do
        if [ "$n" -lt "$c" ] || [ -e "runs/$RUN/.remote_done" ] || [ -e "runs/$RUN/.pod_gone" ]; then
            measure "$n"
            # Between measurements, release a finished pod first: it bills while it waits.
            if [ ! -e "runs/$RUN/.remote_done" ] && pod_alive \
                && "${RP[@]}" ssh "$POD" -- test -e /root/taz/work/DONE 2> /dev/null; then
                finish_pod
            fi
        fi
    done

    pending=0
    for n in $JUDGED; do [ -e "runs/$RUN/.measured_$(printf %04d "$n")" ] || pending=1; done
    if [ $pending = 0 ] && { [ -e "runs/$RUN/.remote_done" ] || [ -e "runs/$RUN/.pod_gone" ]; }; then
        log "campaign complete"
        break
    fi
    [ -e "runs/$RUN/.pod_gone" ] && [ $pending = 1 ] && log "pod gone; $pending judged checkpoint(s) cannot be measured" && break
    sleep "$PULL_EVERY"
done
