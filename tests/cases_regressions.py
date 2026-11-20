#!/usr/bin/env python3
"""Host regressions for second-pass case-study fixes.

Requires NumPy. Does not import torch, TVM, Triton or use a GPU. For torch
checks it executes the actual assertion AST with a NumPy assertion adapter;
this tests fail-fast wiring, not framework compiler execution or torch internals.
"""
from pathlib import Path
import ast
import importlib.util
import subprocess
import sys
import types
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
checks = 0

def load(prefix):
    path = next((ROOT/'docs/part6/code').glob(prefix+'_*.py'))
    spec = importlib.util.spec_from_file_location('book_'+prefix, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

quant = load('078')
for qlo, qhi in ((-128,127), (1,127), (-128,-1)):
    x = np.zeros(2)
    scale, zero = quant.choose_scale_zero_point(x, qlo, qhi)
    assert qlo <= zero <= qhi
    restored = quant.dequantize(quant.quantize(x, scale, zero, qlo, qhi), scale, zero)
    np.testing.assert_array_equal(restored, x)
    checks += 1
print('PASS calibration: default, positive-only and negative-only quantized subranges')

# Exercise canonical numerical-acceptance statements on deliberate mismatches.
# NumPy is the actual implementation for TVM result checks; the tiny torch
# namespace adapter preserves the explicit atol/rtol/equal_nan arguments.
for prefix in ('066','067','068','072','073','074','077','080'):
    path = next((ROOT/'docs/part6/code').glob(prefix+'_*.py'))
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)
             and ast.unparse(node.value.func) in
             ('torch.testing.assert_close', 'np.testing.assert_allclose')]
    assert calls, f'{path.name} has no numerical acceptance assertion'
    for node in calls:
        code = ast.unparse(node)
        call = node.value
        assert all(isinstance(x, ast.Name) for x in call.args[:2])
        first, second = [x.id for x in call.args[:2]]
        kwargs = {k.arg for k in call.keywords}
        assert {'atol','rtol','equal_nan'} <= kwargs, code
        env = {'np':np, 'torch':types.SimpleNamespace(testing=types.SimpleNamespace(
            assert_close=np.testing.assert_allclose)),
            first:np.array([0.,1.]),second:np.array([0.,1.])}
        exec(compile(ast.Module(body=[node],type_ignores=[]), str(path), 'exec'), env)
        preamble = ('import numpy as np, types\n'
          'torch=types.SimpleNamespace(testing=types.SimpleNamespace(assert_close=np.testing.assert_allclose))\n'
          f'{first}=np.array([1000.,1000.])\n{second}=np.array([0.,0.])\n')
        failed = subprocess.run([sys.executable,'-c',preamble+code],capture_output=True,text=True)
        assert failed.returncode != 0, f'{path.name}: mismatch did not fail'
        checks += 1
    print(f'PASS {path.name}: {len(calls)} assertion(s), explicit tolerances and nonzero mismatch exit')
print(f'ALL {checks} host case-study regression checks PASSED; no framework/GPU execution claimed')
