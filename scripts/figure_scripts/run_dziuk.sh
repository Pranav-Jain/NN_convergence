#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface dziuk --pde helmholtz --bc dirichlet --k -1 --experiment montecarlo --size 150 --layers 3 --example 1 --render
