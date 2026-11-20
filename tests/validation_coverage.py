#!/usr/bin/env python3
"""Regression: validators must reject missing/empty/incomplete input coverage."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('coverage_manifest',ROOT/'tools/validation_manifest.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
manifest=module.load_manifest(ROOT)
assert len(module.require_inputs(ROOT,'host'))==78
assert len(module.require_inputs(ROOT,'gpu'))==3
assert len(module.require_inputs(ROOT,'pages'))==43
assert len(module.require_inputs(ROOT,'sources'))==108
checks=0
with tempfile.TemporaryDirectory(prefix='hammer-coverage-regression-') as tmp:
 root=Path(tmp);(root/'tools').mkdir();(root/'bin').mkdir()
 for name in ['validate_gpu.py','validate_host.py','verify_book.py','verify_foundations.py','validation_manifest.py']:
  shutil.copyfile(ROOT/'tools'/name,root/'tools'/name)
 for name in ['nvcc','nvidia-smi']:
  path=root/'bin'/name;path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
 env=os.environ.copy();env['PATH']=str(root/'bin')+os.pathsep+env['PATH']
 def rejected(name,reason):
  global checks
  r=subprocess.run([sys.executable,str(root/'tools'/name)],env=env,text=True,capture_output=True)
  assert r.returncode!=0,(name,reason,r.stdout,r.stderr)
  assert 'passed' not in r.stdout.lower() and 'PASS:' not in r.stdout,(name,r.stdout)
  print('PASS rejected:',name,reason);checks+=1
 for name in ['validate_gpu.py','validate_host.py','verify_book.py','verify_foundations.py']:rejected(name,'missing book manifest')
 (root/'tools/book-manifest.json').write_text(json.dumps(manifest))
 for name in ['validate_gpu.py','validate_host.py','verify_book.py','verify_foundations.py']:rejected(name,'missing docs tree')
 (root/'docs').mkdir()
 for name in ['validate_gpu.py','validate_host.py','verify_book.py','verify_foundations.py']:rejected(name,'empty docs tree')
 for field in ['pages','sources']:
  for relative in manifest[field]:
   path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
 # A single missing expected driver/lesson must reject partial discovery.
 missing=root/next(iter(manifest['gpu_drivers']));missing.unlink()
 rejected('validate_gpu.py','one expected CUDA driver missing');missing.touch()
 missing=root/next(p for p in manifest['sources'] if p.endswith('.cpp'));missing.unlink()
 rejected('validate_host.py','one expected host lesson missing');rejected('verify_foundations.py','one expected host lesson missing');missing.touch()
 rejected('verify_book.py','missing required diagram manifest')
 (root/'tools/diagram-manifest.json').write_text('[]')
 rejected('verify_book.py','empty diagram manifest')
print(f'{checks} negative coverage cases passed; real manifest counts are 78 host / 3 CUDA / 43 pages / 108 sources.')
