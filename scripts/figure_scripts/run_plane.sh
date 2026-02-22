#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface plane --pde poisson --bc dirichlet --experiment interpolation --size 150 --layers 3 --example 1 --render
