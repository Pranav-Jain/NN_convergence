#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface 2d --pde poisson --bc dirichlet --experiment comparison --size 150 --layers 3 --example 10 --render-only
