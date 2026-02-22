#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface springer --pde poisson --bc dirichlet --size 30 --layers 3 --example 3 --copy-plot --render
python3 "$ROOT/scripts/run_mesh_pipeline.py" --surface springer --pde poisson --bc dirichlet --size 150 --layers 3 --example 3 --render
