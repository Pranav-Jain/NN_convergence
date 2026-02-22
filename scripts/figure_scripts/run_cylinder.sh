#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface cylinder --pde poisson --bc dirichlet --experiment minimal_surface --size 100 --layers 3 --example 1 --render
