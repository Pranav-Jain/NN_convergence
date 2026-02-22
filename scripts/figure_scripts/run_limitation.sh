#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN withNN --size 30 --layers 3 --example 10 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN withNN --size 90 --layers 3 --example 10 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface heightfield --pde poisson --bc neumann --NN withNN --size 150 --layers 3 --example 10 --render

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface ellipsoid --pde poisson --bc dirichlet --NN noNN --size 30 --layers 3 --example 12 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface ellipsoid --pde poisson --bc dirichlet --NN noNN --size 90 --layers 3 --example 12 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface ellipsoid --pde poisson --bc dirichlet --NN noNN --size 150 --layers 3 --example 12 --render