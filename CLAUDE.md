# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Experiments using [NetKet](https://www.netket.org/) (a JAX-based library for neural-network quantum
states) to train a Vision Transformer (ViT) variational ansatz for the ground state of the 2D J1-J2
Heisenberg model, run as SLURM GPU jobs on a cluster. Everything currently lives under `vit-testing/`.

## Environment

- Python venv at `.venv/`. Activate with `source .venv/bin/activate`.
- There is no `requirements.txt` / `pyproject.toml` — dependencies (netket, jax + jax-cuda12 plugins,
  flax, einops, matplotlib, jaxtyping) were installed directly into `.venv`. If you add a new import,
  install it into `.venv` with pip; there's no manifest to update.
- `pytest` is not currently installed in `.venv`, even though `test_vit_model.py` uses pytest-style
  `test_*` functions. Install it (`pip install pytest`) before trying to run the tests.

## Common commands

Run the smoke tests for the ViT model building blocks:
```
source .venv/bin/activate
pytest vit-testing/test_vit_model.py
```
Run a single test: `pytest vit-testing/test_vit_model.py::test_vit_shape`

Run a training job locally (CPU or whatever `jax` picks up):
```
source .venv/bin/activate
python vit-testing/netket-vit.py --J2 0.1 --L 10
```
See `parse_args()` in `netket-vit.py` for the full set of hyperparameter flags (lattice size, ViT
architecture, sampler, optimizer, divergence threshold, etc).

Submit as a SLURM GPU job:
```
sbatch vit-testing/netket-vit.slurm --J2 0.1
# override the GPU type per-submission, e.g.:
sbatch --gres=gpu:A100.40gb:1 vit-testing/netket-vit.slurm --J2 0.1
sbatch --gres=gpu:3g.40gb:1   vit-testing/netket-vit.slurm --J2 2 --diag-shift 1e-3
```
The `.slurm` script hardcodes the absolute path `/home/tcosgrov/code/netket-testing` to source the venv
and locate `netket-vit.py` — update both paths together if the repo is ever relocated.

## Architecture

- `vit_model.py` — the ViT ansatz itself, as flax `nn.Module`s:
  - `extract_patches2d` / `Embed` — reshapes a flat spin configuration (length `Ns`) into a square grid
    of `patch_size x patch_size` patches and linearly embeds each patch.
  - `FMHA` (factored multi-head attention) — attention over patches using a learned `alpha` weighting
    instead of query/key dot-products. When `transl_invariant=True`, `alpha` is defined on one head's
    worth of shift offsets and expanded to the full patch-by-patch matrix via `roll2d`, so the same
    attention pattern applies at every patch position (translation invariance from the enlarged
    Hypercube lattice's periodic boundary conditions).
  - `EncoderBlock` / `Encoder` — pre-norm transformer blocks (attn + MLP) stacked `num_layers` times.
  - `OutputHead` — pools over patches, produces separate real/imaginary heads, combines them into a
    complex log-amplitude via `log_cosh` (this is a log-wavefunction ansatz, not a classifier).
  - `ViT` — wires the above together; the `nn.compact` forward pass takes a batch of spin
    configurations directly (shape `(batch, Ns)` of ±1) and returns `log_psi` per sample.
  - All dense/norm layers use `param_dtype = jnp.float64` — NetKet's VMC/QGT machinery expects
    double-precision parameters; don't silently drop to float32 when touching this file.
- `netket-vit.py` — the training entrypoint. Builds the `Hypercube` lattice + `Heisenberg` Hamiltonian
  (J1-J2, `sign_rule=[False, False]`, i.e. no Marshall sign rule), instantiates the `ViT` model,
  `MetropolisExchange` sampler, and runs NetKet's `VMC_SR` (stochastic-reconfiguration VMC) driver.
  - `divergence_guard` is a driver callback that halts the run early if `|energy per site|` exceeds
    `--divergence-threshold`; the frustrated `J2=2` regime is known to occasionally diverge, and this
    avoids burning the rest of a GPU allocation on a run that's already blown up.
  - Runs are logged via `nk.logging.JsonLog` (raw energy history + periodic parameter checkpoints
    every 50 steps) rather than kept only in memory, so a run can be reanalyzed/resumed without
    rerunning the GPU job. Output files are named `J2-{J2}_L-{L}_seed-{seed}` under `--output-dir`.
  - `NETKET_EXPERIMENTAL_SHARDING=0` is required (set in the `.slurm` script) — sharding mode only
    supports jax operators for `get_conn_padded`, and this pipeline hits that path.
- `test_vit_model.py` — shape-only smoke tests for `Embed`, `FMHA`, `Encoder`, and `ViT` on a 10x10
  lattice; they check tensor shapes through the pipeline, not numerical correctness.
- `*.png`, `gpu_job-*.{out,err}` under `vit-testing/` are artifacts from past SLURM runs (plots of
  energy-per-site vs. iteration, and job stdout/stderr) — reference material, not something to edit.
