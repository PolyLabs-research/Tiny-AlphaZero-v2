#!/bin/bash
# Prepare a fresh pod (runpod/pytorch:...-torch2121-ubuntu2404) to run this repo:
#
#   bash setup_pod.sh /root/taz/chess.bundle
#
# Clones the code from a git bundle pushed from this box (the pod never holds GitHub
# credentials), installs the project against the image's own torch -- --no-deps, so
# pip cannot swap in another build -- installs Stockfish from apt (Ubuntu 24.04 ships
# 16-1build1, the skill-0 instrument's build), and prints the stack it ended up on.
set -euo pipefail
# An SSH session does not inherit the container's environment -- the image's PIP_*
# settings, its CUDA PATH, RunPod's injected RUNPOD_* -- so adopt pid 1's.
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ
BUNDLE=${1:-/root/taz/chess.bundle}
cd /root
# A bundle made with `git bundle create X main` carries no HEAD, so name the branch.
rm -rf chess
git clone -q -b main "$BUNDLE" chess
cd chess
python3 -m pip install -q --no-deps -e .
python3 -m pip install -q "numpy>=1.26" "chess>=1.10" "pyyaml>=6.0"
DEBIAN_FRONTEND=noninteractive apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq install -y stockfish rsync > /dev/null
quota=$(cat /sys/fs/cgroup/cpu.max 2> /dev/null ||
    echo "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)")
echo "code:      $(git rev-parse --short HEAD)"
echo "stack:     $(python3 -c 'import sys, torch; print(sys.version.split()[0], torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))')"
echo "stockfish: $(dpkg-query -W -f='${Version}' stockfish) $(command -v stockfish || echo /usr/games/stockfish)"
echo "cpu:       quota '$quota'  RUNPOD_CPU_COUNT=${RUNPOD_CPU_COUNT:-?}  RAM ${RUNPOD_MEM_GB:-?} GB"
