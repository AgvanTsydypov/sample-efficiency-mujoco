#!/usr/bin/env bash
#
# Run the full PPO vs SAC sweep with a bounded number of concurrent processes.
#
# Usage:
#   ./run_all.sh                           # 3 seeds, Hopper-v5, 1M steps
#   SEEDS=5 ./run_all.sh                   # 5 seeds
#   ENV=Walker2d-v5 STEPS=2000000 ./run_all.sh
#
set -euo pipefail

ENV="${ENV:-Hopper-v5}"
STEPS="${STEPS:-1000000}"
SEEDS="${SEEDS:-3}"

# One BLAS thread per process. The networks here are two 256-unit layers, far
# too small for multithreaded matrix ops to pay off, and MuJoCo's own step is
# single threaded. Parallelism belongs across runs, not inside them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1   # Apple Accelerate
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Concurrency defaults to the number of performance cores. Efficiency cores are
# roughly half as fast, so a run scheduled onto one becomes the straggler that
# sets total wall clock time.
if [[ "$(uname)" == "Darwin" ]]; then
    P_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu)
else
    P_CORES=$(nproc)
fi
JOBS="${JOBS:-$P_CORES}"

mkdir -p run_logs

echo "Environment:  $ENV"
echo "Steps/run:    $STEPS"
echo "Seeds:        $SEEDS"
echo "Concurrency:  $JOBS  (performance cores detected: $P_CORES)"
echo

# Build the job list, then feed it to xargs as a bounded queue. As soon as one
# run finishes the next starts, so no core sits idle waiting for a slow job.
JOB_LIST=$(mktemp)
trap 'rm -f "$JOB_LIST"' EXIT

for algo in ppo sac; do
    for ((seed = 0; seed < SEEDS; seed++)); do
        echo "$algo $seed"
    done
done > "$JOB_LIST"

run_one() {
    local algo="$1" seed="$2"
    local log="run_logs/${algo}_seed${seed}.log"
    echo "start  $algo seed $seed"
    if python run_experiment.py --algo "$algo" --seed "$seed" \
        --env "$ENV" --steps "$STEPS" > "$log" 2>&1; then
        echo "done   $algo seed $seed"
    else
        echo "FAILED $algo seed $seed  (see $log)"
    fi
}
export -f run_one
export ENV STEPS

START=$(date +%s)

# caffeinate keeps macOS from sleeping mid sweep. On Linux it is absent and the
# command runs directly.
RUNNER=(xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"')
if command -v caffeinate > /dev/null; then
    caffeinate -i xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"' < "$JOB_LIST"
else
    "${RUNNER[@]}" < "$JOB_LIST"
fi

ELAPSED=$(( $(date +%s) - START ))
echo
echo "Sweep finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo

python aggregate_results.py --env "$ENV"
