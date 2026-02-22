#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface hammer --pde poisson --bc dirichlet --size 150 --layers 3 --example 1 --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface hammer --pde poisson --bc dirichlet --size 150 --layers 3 --example 1_Niloy --render-only
