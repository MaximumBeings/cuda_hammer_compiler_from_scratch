#!/usr/bin/env python3
"""Run Chapter18 generators and child kernels on a real CUDA GPU. A10: --arch sm_86."""
import argparse, pathlib, re, shutil, subprocess, sys
from validation_manifest import CoverageError, load_manifest, require_inputs
p=argparse.ArgumentParser(); p.add_argument('--arch',default='sm_86'); p.add_argument('--sanitize',action='store_true'); a=p.parse_args()
if not re.fullmatch(r'sm_[0-9]+',a.arch): p.error('expected sm_NN')
root=pathlib.Path(__file__).resolve().parents[1]
try:
 sources=require_inputs(root,'gpu'); driver_programs=load_manifest(root)['gpu_drivers']
except CoverageError as ex: sys.exit('Coverage failure: '+str(ex))
logs=root/'validation/accuracy-fixes/gpu';logs.mkdir(parents=True,exist_ok=True)
if not shutil.which('nvcc'): sys.exit('nvcc unavailable: no GPU validation performed')
def run(cmd,path):
 r=subprocess.run(cmd,cwd=root,text=True,errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT);path.write_text('COMMAND: '+repr(cmd)+'\n'+r.stdout+'\nEXIT: '+str(r.returncode)+'\n');return r
info=run(['nvidia-smi'],logs/'nvidia-smi.txt')
if info.returncode: sys.exit('GPU unavailable: no runtime validation performed')
failed=False
for src in sources:
 exe=logs/(src.stem+'.bin')
 c=run(['nvcc','-std=c++17','-O2','-arch='+a.arch,'-DHAMMER_CUDA_ARCH="'+a.arch+'"',str(src),'-o',str(exe)],logs/(src.stem+'-compile.txt'))
 if c.returncode: failed=True;continue
 r=run([str(exe)],logs/(src.stem+'-run.txt'))
 if r.returncode == 77: print(src.name+': SKIP(no physical CUDA device); runtime validation incomplete')
 if r.returncode or re.search(r'no CUDA.capable|MISMATCH|FAILED|CUDA error',r.stdout,re.I):
  print(src.name,'runtime validation failed',r.returncode,flush=True); failed=True
 if a.sanitize:
  if not shutil.which('compute-sanitizer'): sys.exit('compute-sanitizer requested but unavailable')
  # Instrument the generated CUDA programs themselves. This also works on
  # sanitizer builds that do not follow the host-only driver's popen/exec tree.
  generated = driver_programs[str(src.relative_to(root))]
  for program in generated:
   for tool in ['memcheck','racecheck','synccheck']:
    r=run(['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','99',program],logs/(pathlib.Path(program).name+'-'+tool+'.txt'))
    summary_ok = re.search(r'(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)',r.stdout) is not None
    if r.returncode or not summary_ok:
     print(program,tool,"failed",r.returncode,"summary present",summary_ok,flush=True); failed=True
print('FAILED' if failed else 'Chapter18 GPU runtime checks passed; inspect logs for device-specific results.')
sys.exit(1 if failed else 0)
