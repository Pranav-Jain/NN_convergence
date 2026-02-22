#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 30 --layers 3 --example 5 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 90 --layers 3 --example 5 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 150 --layers 3 --example 5 --render

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 90 --layers 2 --example 5 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 90 --layers 5 --example 5 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface mushroom --pde poisson --bc dirichlet --size 90 --layers 8 --example 5 --render
