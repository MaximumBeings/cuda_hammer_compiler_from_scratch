from pathlib import Path
import ast, importlib.util, types
import numpy as np
ROOT=Path('/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch')
p=next((ROOT/'docs/part6/code').glob('078_*.py'))
spec=importlib.util.spec_from_file_location('quant',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
print('Calibration fallback:')
for lo,hi in [(-128,127),(1,127),(-128,-1)]:
 s,z=mod.choose_scale_zero_point(np.zeros(2),lo,hi)
 print('  range',lo,hi,'returned scale',s,'zero_point',z,end='; ')
 try:print('quantized',mod.quantize(np.zeros(2),s,z,lo,hi))
 except ValueError as e:print(type(e).__name__+':',e)
print('\nExact comparison-print AST extracted from canonical File072:')
p=next((ROOT/'docs/part6/code').glob('072_*.py'));tree=ast.parse(p.read_text())
node=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id',None)=='print' and any(isinstance(x,ast.Constant) and isinstance(x.value,str) and 'Real correctness check: compiled vs.' in x.value for x in ast.walk(n)))
test=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(test)
exec(compile(test,str(p),'exec'),{'torch':types.SimpleNamespace(allclose=np.allclose),'out':np.array([1000.]),'eager_out':np.array([0.])})
print('Comparison mismatch raised no exception; full source contains',sum(isinstance(n,ast.Assert) for n in ast.walk(tree)),'assert statements.')
