#!/usr/bin/env bash
# Build + smoke-test all kernel versions, then point the user at the
# benchmark + plot pipeline. Run from the flashinfer_engine/ directory.
set -euo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/2] Building flashattn_v0 kernels (prefill + decode)"
(cd "${ENGINE_DIR}" && python -m csrc.compile)

echo "[2/2] Running smoke tests"
(cd "${ENGINE_DIR}" && pytest tests/test_smoke.py -v)

echo ""
echo "v0 kernel build + validation complete."
echo "Next: run benchmarks and plot results:"
echo "  python benchmarks/benchmark_prefill.py --model configs/models/llama3_2_1b.yaml --backends sdpa flashattn_v0"
echo "  python benchmarks/benchmark_decode.py  --model configs/models/llama3_2_1b.yaml --backends sdpa_naive flashattn_v0"
echo "  python benchmarks/plot_results.py"
