#!/bin/sh
# Follow-up hyperparameter probe for the J2>=1.6 regime.
#
# Usage: ./submit_hparam_probe2.sh

set -eu

REPO=/home/tcosgrov/code/netket-testing
OUTDIR="$REPO/vit-testing/hparam-probe2"
mkdir -p "$OUTDIR"

J2_VALUES="1.7 2.0"
SEEDS="0 2"

# All configs build on diag-shift, since 1e-3 was clearly directionally
# right; the question is whether more of it (and/or a smaller step) is
# enough to hold at high J2.
#   F: 3e-3                        -- 3x the current best
#   G: 1e-2                        -- 10x the current best
#   H: 1e-3 + lr 0.003             -- both first-probe winners combined
#   I: 3e-3 + lr 0.003             -- more regularization AND smaller steps
CONFIGS='
F:--diag-shift 3e-3:
G:--diag-shift 1e-2:
H:--diag-shift 1e-3 --learning-rate 0.003:
I:--diag-shift 3e-3 --learning-rate 0.003:
'

for J2 in $J2_VALUES; do
  for SEED in $SEEDS; do
    echo "$CONFIGS" | while IFS=: read -r LABEL FLAGS _; do
      [ -z "$LABEL" ] && continue
      sbatch "$REPO/vit-testing/netket-vit.slurm" \
        --J2 "$J2" \
        --seed "$SEED" \
        --n-iter 1500 \
        $FLAGS \
        --output-dir "$OUTDIR"
    done
  done
done
