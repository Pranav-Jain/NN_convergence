#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface moai --pde helmholtz --bc dirichlet --experiment heat --k -100 --size 150 --layers 3 --example 13_100 --render-only
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface moai --pde helmholtz --bc dirichlet --experiment heat --k -250 --size 150 --layers 3 --example 13_250 --render-only
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface moai --pde helmholtz --bc dirichlet --experiment heat --k -500 --size 150 --layers 3 --example 13_500 --render-only
