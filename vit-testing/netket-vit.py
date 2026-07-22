import argparse
import os
import sys
import time

import jax
import matplotlib.pyplot as plt
import netket as nk

# Variational monte carlo driver
from netket.driver import VMC_SR

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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=10.0,
        help=(
            "Stop the run early if |energy per site| exceeds this value. "
            "Converged runs typically sit around O(1), so this catches "
            "blow-ups (seen e.g. in the frustrated J2=2 regime) well before "
            "they reach extreme values, without wasting the rest of the "
            "GPU allocation on a diverged run."
        ),
    )
    return parser.parse_args()


args = parse_args()

start_time = time.perf_counter()

key = jax.random.key(args.seed)

L = args.L
n_dim = args.n_dim
# J2 / J1 => Go from J2 = 0.1 to J2 = 1
J2 = args.J2

os.makedirs(args.output_dir, exist_ok=True)
run_name = f"J2-{args.J2}_L-{args.L}_seed-{args.seed}"

lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

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

# Metropolis Local Sampling
N_samples = args.n_samples
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=args.sweep_d_max,
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


vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=args.diag_shift,
    variational_state=vstate,
    mode="complex",
)
# Optimization. JsonLog persists the raw energy history (and periodic
# parameter checkpoints) to disk as the run progresses, so results can be
# reanalyzed or resumed from without rerunning the (expensive) GPU job.
log = nk.logging.JsonLog(
    os.path.join(args.output_dir, run_name), save_params=True, save_params_every=50
)


def divergence_guard(step, logged_data, driver):
    energy_per_site = logged_data["Energy"]["Mean"].real / (L * L * 4)
    if abs(energy_per_site) > args.divergence_threshold:
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
