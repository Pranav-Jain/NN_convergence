#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface 2d --pde poisson --bc dirichlet --experiment activation --size 150 --layers 3 --example 9_ours --render-only
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface 2d --pde poisson --bc dirichlet --experiment activation --size 150 --layers 3 --example 9_sigmoid --render-only
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface 2d --pde poisson --bc dirichlet --experiment activation --size 150 --layers 3 --example 9_tanh --render-only