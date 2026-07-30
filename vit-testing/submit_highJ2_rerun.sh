#!/bin/sh
#
# Usage: ./submit_highJ2_rerun.sh   (run from anywhere; paths below are absolute)

set -eu

REPO=/home/tcosgrov/code/netket-testing
OUTDIR="$REPO/vit-testing/j2-sweep-L10"
mkdir -p "$OUTDIR"

J2_VALUES="1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
SEEDS="0 1 2 3"

for J2 in $J2_VALUES; do
  for SEED in $SEEDS; do
    sbatch "$REPO/vit-testing/netket-vit.slurm" \
      --J2 "$J2" \
      --seed "$SEED" \
      --n-iter 3000 \
      --diag-shift 1e-3 \
      --output-dir "$OUTDIR"
  done
done
