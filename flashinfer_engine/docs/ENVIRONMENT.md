# Environment and Compatibility

## Baseline Runtime

- Python: 3.12
- PyTorch: CUDA-enabled build
- CUDA module used during successful runs: 13.x family

## Known Compatibility Notes

1. **CUDA mismatch**  
   If compile fails with mismatch errors, verify:
   - `torch.version.cuda`
   - `nvcc --version`
   - loaded CUDA module version

2. **Driver too old on some nodes**  
   Some GPU nodes (for example, certain RTX 6000 nodes) may have drivers too old for the current PyTorch CUDA build.
   Symptom: `torch._C._cuda_init()` fails before tests start.

3. **GPU resource naming in Slurm**  
   Request exact resource names from `sinfo` output (example: `rtx_6000`, not `RTX6000`).

## Recommended Preflight

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
nvcc --version
nvidia-smi
```
