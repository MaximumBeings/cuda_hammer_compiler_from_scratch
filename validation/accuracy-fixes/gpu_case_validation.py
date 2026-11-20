"""Execute the book's real Triton kernels; requires CUDA torch and Triton.

Run from the repository root on an NVIDIA GPU, not TRITON_INTERPRET=1.
An offline compile or missing GPU is never recorded as a passed execution.
"""
from pathlib import Path
import importlib.util
import sys
import torch
import triton

ROOT = Path(__file__).resolve().parents[2]

def load(prefix):
    path = next((ROOT / 'docs/part6/code').glob(prefix + '*.py'))
    spec = importlib.util.spec_from_file_location('book_' + prefix, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

if not torch.cuda.is_available():
    raise SystemExit('FAIL: a real visible CUDA GPU is required')
print('torch', torch.__version__, 'triton', triton.__version__)
print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
torch.manual_seed(2026)
attn = load('076_')
quant = load('079_')
checks = 0
for rows, keys, dim, tile in [(16, 1, 16, 16), (16, 17, 16, 16),
                              (16, 31, 16, 16), (16, 48, 16, 16),
                              (32, 49, 32, 32), (16, 65, 64, 32)]:
    q = torch.randn(rows, dim, device='cuda', dtype=torch.float32)
    k = torch.randn(keys, dim, device='cuda', dtype=torch.float32)
    v = torch.randn(keys, dim, device='cuda', dtype=torch.float32)
    out = torch.empty_like(q)
    attn.flash_attention_kernel[(1,)](q, k, v, out, BLOCK_M=rows,
        BLOCK_N=tile, D=dim, SEQ_LEN_KV=keys)
    torch.cuda.synchronize()
    # Independent float64 reference avoids TF32-default matmul effects.
    ref = (torch.softmax((q.double() @ k.double().T) / dim**0.5, dim=-1)
           @ v.double()).float()
    torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)
    assert torch.isfinite(out).all()
    checks += 1
    print('PASS attention', (rows, keys, dim, tile), 'max_abs', (out-ref).abs().max().item())
for n in (32, 64, 128):
    for extreme in (False, True):
        a = torch.full((n,), -128, device='cuda', dtype=torch.int8) if extreme else torch.randint(-128, 128, (n,), device='cuda', dtype=torch.int8)
        b = torch.full_like(a, -128) if extreme else torch.randint(-128, 128, (n,), device='cuda', dtype=torch.int8)
        out = torch.empty((), device='cuda', dtype=torch.float32)
        quant.quantized_dot_kernel[(1,)](a, b, out, 0.125, 0.0625, N=n)
        torch.cuda.synchronize()
        ref = ((a.long()*b.long()).sum().double()*0.125*0.0625).float()
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-6)
        checks += 1
        print('PASS int8 dot', n, 'extreme', extreme, 'result', out.item())
print('ALL', checks, 'GPU execution checks PASSED')
