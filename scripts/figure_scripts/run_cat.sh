#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface cat --pde helmholtz --bc dirichlet --k -20 --experiment heat --size 150 --layers 3 --example 11 --render
