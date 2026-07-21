#SBATCH --job-name=netket-vit-testing
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
## Separate output and error messages into 2 files.
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file

## Disable email notifications
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tcosgrov@gmu.edu

## Time the job needs. (default: see partition above)
#SBATCH --time=0-01:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # number of cores needed
#SBATCH --gres=gpu:1g.10gb:1          # here you should insert the total number of gpus per node
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=nomultithread  # Disable Hyperthreading

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

## Load the relevant modules needed for the job
module load gnu/12.3.0
module load python/3.12.1-33

## Source the venv
source /home/tcosgrov/code/netket-testing/.venv/bin/activate

# Tell NetKet to NOT use sharding mode.
# Otherwise you get this: 
# File "/home/tcosgrov/code/mb-psi-net/.venv/lib/python3.12/site-packages/netket/operator/_discrete_operator.py", line 69, in get_conn_padded
# RuntimeError: When using Sharding mode, only jax operators are supported.
export NETKET_EXPERIMENTAL_SHARDING=0
# Tell Jax that we want to use GPUs. This is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

srun -uv python3 /home/tcosgrov/code/netket-testing/netket-vit.py