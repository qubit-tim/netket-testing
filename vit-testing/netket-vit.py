"""Train a ViT variational wave function on the J1-J2 Heisenberg model via SR-VMC.

Run once per (J2, L, seed); normally submitted as a SLURM job (netket-vit.slurm).
"""

import argparse
import math
import os
import sys
import time

import jax
import matplotlib.pyplot as plt
import netket as nk

# Variational monte carlo driver
from netket.driver import VMC_SR

from netket.operator.spin import sigmax, sigmay, sigmaz

from vit_model import ViT

print(f"Python Version == {sys.version}")
print(jax.devices())

# Loop through all active modules in the runtime environment
for name, module in sorted(sys.modules.items()):
    if hasattr(module, "__version__"):
        print(f"{name} == {module.__version__}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ViT variational wave function on the J1-J2 Heisenberg model."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--L", type=int, default=10, help="Linear lattice size")
    parser.add_argument("--n-dim", type=int, default=2)
    parser.add_argument(
        "--J2", type=float, default=2, help="J2/J1 ratio (0.1 to 1 is the typical sweep)"
    )
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=60)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument(
        "--transl-invariant", dest="transl_invariant", action="store_true", default=True
    )
    parser.add_argument(
        "--no-transl-invariant", dest="transl_invariant", action="store_false"
    )
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--sweep-d-max", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0075)
    parser.add_argument("--diag-shift", type=float, default=1e-4)
    parser.add_argument(
        "--linear-solver",
        type=str,
        choices=["cholesky_with_fallback", "cholesky", "pinv_smooth"],
        default="cholesky_with_fallback",
        help=(
            "Linear solver for the SR update. 'cholesky' is fastest but most "
            "prone to NaN blow-ups; 'pinv_smooth' is more stable but costlier "
            "per step; the default falls back to pinv_smooth on NaN/Inf."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default=".")
    def _total_sz(value):
        return None if value.strip().lower() == "none" else float(value)

    parser.add_argument(
        "--total-sz",
        type=_total_sz,
        default=0.0,
        help=(
            "Total magnetization sector to restrict to, or 'none' for an "
            "unrestricted Hilbert space. Also selects the sampler: a fixed "
            "sector uses spin-exchange moves, 'none' uses single-spin flips."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["heisenberg", "xy", "ising"],
        default="heisenberg",
        help="Which spin couplings to use on each bond; all share the --J2 ratio.",
    )
    parser.add_argument(
        "--field",
        type=float,
        default=0.0,
        help=(
            "Longitudinal field strength, added as -field * sum_i Sz_i. "
            "Requires --total-sz none; in a fixed sector it is a no-op."
        ),
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=10.0,
        help=(
            "Stop early if |energy per site| exceeds this. Converged runs sit "
            "around O(1), so this catches blow-ups without burning the rest "
            "of the GPU allocation."
        ),
    )
    return parser.parse_args()


args = parse_args()

# sum_i Sz_i is constant within a fixed total_sz sector, so the field term
# would be a no-op rather than an error. Fail loudly instead.
if args.field != 0.0 and args.total_sz is not None:
    raise ValueError(
        "--field requires --total-sz none; in a fixed sector it does nothing."
    )

start_time = time.perf_counter()

key = jax.random.key(args.seed)

L = args.L
n_dim = args.n_dim
# J2 / J1 => Go from J2 = 0.1 to J2 = 1
J2 = args.J2

os.makedirs(args.output_dir, exist_ok=True)
run_name = f"J2-{args.J2}_L-{args.L}_seed-{args.seed}"
# Only append non-default settings, so default-config filenames stay stable
# across runs (and differently-tuned runs don't overwrite each other).
if args.model != "heisenberg":
    run_name += f"_model-{args.model}"
if args.total_sz != 0.0:
    run_name += f"_totalsz-{args.total_sz}"
if args.field != 0.0:
    run_name += f"_field-{args.field}"
if args.linear_solver != "cholesky_with_fallback":
    run_name += f"_solver-{args.linear_solver}"
if args.learning_rate != 0.0075:
    run_name += f"_lr-{args.learning_rate}"
if args.diag_shift != 1e-4:
    run_name += f"_diagshift-{args.diag_shift}"

# max_neighbor_order=2 exposes both the J1 (color 0) and J2 (color 1) bonds.
# pbc=True is also what makes the lattice translation-invariant, which the
# --transl-invariant attention in vit_model.py assumes; don't disable one
# without the other.
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# total_sz=0 (the default) is the zero-magnetization sector the antiferromagnetic
# ground state lives in; another sector converges to a different state entirely.
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=args.total_sz)

if args.model == "heisenberg":
    # sign_rule=False: the Marshall sign trick assumes an unfrustrated
    # bipartite lattice, which no longer holds once J2 > 0.
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
    )
else:
    # No built-in constructor for XY/Ising, so sum over the colored bonds
    # directly (verified to reproduce nk.operator.Heisenberg exactly when all
    # three spin components are included).
    if args.model == "xy":
        def bond_term(i, j):
            return sigmax(hilbert, i) @ sigmax(hilbert, j) + sigmay(hilbert, i) @ sigmay(hilbert, j)
    elif args.model == "ising":
        def bond_term(i, j):
            return sigmaz(hilbert, i) @ sigmaz(hilbert, j)

    hamiltonian = sum(
        weight * bond_term(i, j)
        for weight, color in [(1.0, 0), (J2, 1)]
        for i, j in lattice.edges(filter_color=color)
    )

if args.field != 0.0:
    hamiltonian = hamiltonian - args.field * sum(
        sigmaz(hilbert, i) for i in range(lattice.n_nodes)
    )

hamiltonian = hamiltonian.to_jax_operator()

# Intiialize the ViT variational wave function
vit_module = ViT(
    num_layers=args.num_layers,
    d_model=args.d_model,
    n_heads=args.n_heads,
    patch_size=args.patch_size,
    transl_invariant=args.transl_invariant,
)

# A dummy batch of spin configurations, sized to the actual lattice, used only
# to trace shapes for parameter initialization.
key, subkey, init_key = jax.random.split(key, 3)
init_spins = jax.random.randint(
    subkey, shape=(1, lattice.n_nodes), minval=0, maxval=1
) * 2 - 1
params = vit_module.init(init_key, init_spins)

N_samples = args.n_samples
# The sampler's move set has to match the Hilbert-space restriction: exchange
# moves conserve total_sz, so they can't explore an unrestricted space (they'd
# silently stay in whichever sector the initial configuration landed in).
if args.total_sz is not None:
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=args.sweep_d_max,
        n_chains=N_samples,
        sweep_size=lattice.n_nodes,
    )
else:
    sampler = nk.sampler.MetropolisLocal(
        hilbert=hilbert,
        n_chains=N_samples,
        sweep_size=lattice.n_nodes,
    )

optimizer = nk.optimizer.Sgd(learning_rate=args.learning_rate)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=args.chunk_size,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)


# mode="complex" is required: without the Marshall sign rule the wavefunction
# isn't guaranteed to be real-representable.
linear_solver = {
    "cholesky_with_fallback": nk.optimizer.solver.cholesky_with_fallback,
    "cholesky": nk.optimizer.solver.cholesky,
    "pinv_smooth": nk.optimizer.solver.pinv_smooth,
}[args.linear_solver]

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=args.diag_shift,
    linear_solver=linear_solver,
    variational_state=vstate,
    mode="complex",
)
# JsonLog persists the energy history and periodic parameter checkpoints, so
# runs can be reanalyzed or resumed without repeating the GPU job.
log = nk.logging.JsonLog(
    os.path.join(args.output_dir, run_name), save_params=True, save_params_every=50
)


def divergence_guard(step, logged_data, driver):
    """Halt the run early if the energy blows up. Returning False stops the driver."""
    energy_mean = logged_data["Energy"].Mean
    energy_per_site = float(jax.device_get(energy_mean).real) / (L * L * 4)
    # isnan is load-bearing: NaN comparisons are always False, so a NaN run
    # would otherwise pass the threshold check and run to completion logging
    # nulls (this happened to 4/96 runs in the L=10 J2 sweep).
    if math.isnan(energy_per_site) or abs(energy_per_site) > args.divergence_threshold:
        print(
            f"Diverging: energy_per_site={energy_per_site:.4f} exceeds "
            f"--divergence-threshold={args.divergence_threshold} at step {step}. "
            "Stopping early.",
            flush=True,
        )
        return False
    return True


N_opt = args.n_iter
vmc.run(n_iter=N_opt, out=log, callback=divergence_guard)

# L*L sites, times 4 to convert NetKet's +-1/2 spin convention to the +-1
# convention energies are usually quoted in.
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")

plotname = os.path.join(args.output_dir, f"{run_name}-vit-run.png")
plt.savefig(plotname)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
