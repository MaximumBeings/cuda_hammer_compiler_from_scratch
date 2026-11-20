from pathlib import Path
import concurrent.futures,subprocess,tempfile
ROOT=Path('/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch');OUT=Path('/tmp/hammer-second-foundations/baseline-sanitized');OUT.mkdir(exist_ok=True)
files=sorted(p for i in range(4) for p in (ROOT/f'docs/part{i}/code').glob('[0-9]*.cpp'))
def run(p):
 with tempfile.TemporaryDirectory(prefix='hammer-re-audit-') as d:
  exe=Path(d)/'lesson';c=subprocess.run(['clang++','-std=c++17','-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all',str(p),'-o',str(exe)],text=True,capture_output=True)
  if c.returncode:result=c.stderr;status='COMPILE FAIL'
  else:
   r=subprocess.run([str(exe)],cwd=d,text=True,capture_output=True,timeout=90);result=r.stdout+r.stderr;status='PASS' if r.returncode==0 else 'FAIL '+str(r.returncode)
  (OUT/(p.stem+'.txt')).write_text(result);return str(p.relative_to(ROOT))+' '+status
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(run,files));(OUT/'summary.txt').write_text('\n'.join(results)+'\n');print('\n'.join(results))
