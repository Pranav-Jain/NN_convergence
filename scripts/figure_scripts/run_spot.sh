#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface spot --pde helmholtz --bc dirichlet --k 4 --size 30 --layers 3 --example 1 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface spot --pde helmholtz --bc dirichlet --k 4 --size 90 --layers 3 --example 1 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface spot --pde helmholtz --bc dirichlet --k 4 --size 150 --layers 3 --example 1 --copy-plot --render

python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface spot --pde helmholtz --bc dirichlet --k 4 --size 150 --layers 3 --example 2 --copy-plot
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface spot --pde helmholtz --bc dirichlet --k 4 --size 150 --layers 3 --example 3 --copy-plot