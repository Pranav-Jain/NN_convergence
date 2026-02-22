#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc dirichlet --NN noNN --size 30 --layers 3 --example 2 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc dirichlet --NN noNN --size 90 --layers 3 --example 2 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc dirichlet --NN noNN --size 150 --layers 3 --example 2 --render

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc dirichlet --NN noNN --size 150 --layers 3 --example 3 --copy-plot
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc dirichlet --NN noNN --size 150 --layers 3 --example 4 --copy-plot

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN noNN --size 30 --layers 3 --example 6 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN noNN --size 90 --layers 3 --example 6 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN noNN --size 150 --layers 3 --example 6 --render

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN noNN --size 150 --layers 3 --example 7 --copy-plot
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN noNN --size 150 --layers 3 --example 8 --copy-plot


