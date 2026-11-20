#!/usr/bin/env python3
"""Compile and execute every standalone host C++ lesson in isolated directories."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,json,os,platform,shutil,subprocess,tempfile,time
from validation_manifest import CoverageError, require_inputs
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=3);parser.add_argument('--cxx',default=os.environ.get('CXX','clang++'));parser.add_argument('--sanitize',action='store_true');parser.add_argument('--log-dir',default='validation/host');args=parser.parse_args()
 try: sources=require_inputs(ROOT,'host')
 except CoverageError as ex: parser.error(str(ex))
 compiler=shutil.which(args.cxx)
 if not compiler:parser.error('host C++ compiler not found')
 logdir=Path(args.log_dir);logdir=logdir if logdir.is_absolute() else ROOT/logdir;logdir.mkdir(parents=True,exist_ok=True)
 def check(source):
  with tempfile.TemporaryDirectory(prefix='hammer-host-') as tmp:
   exe=Path(tmp)/'lesson';flags=['-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all'] if args.sanitize else ['-O2'];command=[compiler,'-std=c++17',*flags,'-Wall','-Wextra',str(source),'-o',str(exe)]
   if platform.system()=='Linux':command+=['-ldl']
   started=time.monotonic();log='SOURCE: '+str(source.relative_to(ROOT))+'\nCOMMAND: '+repr(command)+'\n'
   result={'source':str(source.relative_to(ROOT))}
   try:
    c=subprocess.run(command,cwd=tmp,text=True,errors="backslashreplace",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120);log+='COMPILE:\n'+c.stdout;result['compile_exit']=c.returncode
    if c.returncode==0:
     r=subprocess.run([str(exe)],cwd=tmp,text=True,errors="backslashreplace",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180);log+='\nEXECUTION:\n'+r.stdout;result['run_exit']=r.returncode
   except subprocess.TimeoutExpired:result['timeout']=True;log+='\nTIMEOUT\n'
   result['elapsed_seconds']=round(time.monotonic()-started,3);result['passed']=result.get('compile_exit')==0 and result.get('run_exit')==0
   (logdir/(source.stem+'.txt')).write_text(log);return result
 results=[]
 with ThreadPoolExecutor(max_workers=max(1,min(args.jobs,8))) as pool:
  for f in as_completed([pool.submit(check,p) for p in sources]):
   r=f.result();results.append(r);print(('PASS ' if r['passed'] else 'FAIL ')+r['source'],flush=True)
 report={'platform':platform.platform(),'machine':platform.machine(),'sanitizers':args.sanitize,'generated_children_instrumented':False,'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':sorted(results,key=lambda x:x['source'])}
 (logdir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 passed=sum(x['passed'] for x in results);print(f'{passed}/{len(results)} host lessons passed. Generated child programs are validated by their drivers but are not sanitizer-instrumented by this runner; CUDA kernels and external Python frameworks are not executed.')
 return 0 if passed==len(results) else 1
if __name__=='__main__':raise SystemExit(main())
