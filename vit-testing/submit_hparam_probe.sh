#!/bin/sh
# Usage: ./submit_hparam_probe.sh   (run from anywhere; paths below are absolute)

set -eu

REPO=/home/tcosgrov/code/netket-testing
OUTDIR="$REPO/vit-testing/hparam-probe"
mkdir -p "$OUTDIR"

J2_VALUES="1.3 1.6 2.0"
SEEDS="0 1"

# Config label : extra netket-vit.py flags
CONFIGS='
A::
B:--linear-solver pinv_smooth:
C:--diag-shift 1e-3:
D:--learning-rate 0.003:
E:--linear-solver pinv_smooth --diag-shift 1e-3:
'

for J2 in $J2_VALUES; do
  for SEED in $SEEDS; do
    echo "$CONFIGS" | while IFS=: read -r LABEL FLAGS _; do
      [ -z "$LABEL" ] && continue
      sbatch "$REPO/vit-testing/netket-vit.slurm" \
        --J2 "$J2" \
        --seed "$SEED" \
        --n-iter 500 \
        $FLAGS \
        --output-dir "$OUTDIR"
    done
  done
done
