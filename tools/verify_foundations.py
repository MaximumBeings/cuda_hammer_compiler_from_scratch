#!/usr/bin/env python3
"""Compile/run Chapters1-16 and synchronize their source/output Markdown fences."""
from pathlib import Path
import re, subprocess, tempfile, concurrent.futures, os
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'validation/accuracy-fixes/foundations';LOG.mkdir(parents=True,exist_ok=True)
SOURCES=sorted(p for i in range(4) for p in (ROOT/f'docs/part{i}/code').glob('*.cpp'))
def run(p):
 with tempfile.TemporaryDirectory(prefix='hammer-foundation-') as d:
  exe=Path(d)/'lesson'
  c=subprocess.run([os.environ.get('CXX','clang++'),'-std=c++17','-O2','-Wall','-Wextra',str(p),'-o',str(exe)],capture_output=True,text=True)
  if c.returncode:return p,None,c.stderr
  r=subprocess.run([str(exe)],cwd=d,capture_output=True,text=True,timeout=60)
  (LOG/(p.stem+'.stdout.txt')).write_text(r.stdout)
  (LOG/(p.stem+'.stderr.txt')).write_text(c.stderr+r.stderr)
  return p,r.stdout if r.returncode==0 else None,r.stderr
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(run,SOURCES))
byname={p.name:(p,out,err) for p,out,err in results}
for i in range(4):
 for md in (ROOT/f'docs/part{i}').glob('*.md'):
  s=md.read_text();edits=[]
  for m in re.finditer(r'```cpp\n(.*?)\n```',s,re.S):
   if 'int main(' not in m[1]:continue
   names=re.findall(r'\b\d{3}_[A-Za-z0-9_]+\.cpp\b',m[1])
   name=next((n for n in names if n in byname),None)
   if not name:raise RuntimeError('unmapped main block '+str(md))
   p,out,err=byname[name]
   if out is None:continue
   edits.append((m.start(1),m.end(1),p.read_text().rstrip()))
   tail=s[m.end():];n=re.search(r'\*\*Output:\*\*\s*\n```text\n(.*?)\n```',tail,re.S)
   nxt=tail.find('```cpp')
   if n and (nxt<0 or n.start()<nxt):edits.append((m.end()+n.start(1),m.end()+n.end(1),out.rstrip()))
  for a,b,v in sorted(edits,reverse=True):s=s[:a]+v+s[b:]
  md.write_text(s)
summary='\n'.join(f'{p.relative_to(ROOT)}: '+('PASS' if out is not None else 'FAIL '+err[:300]) for p,out,err in results)
(LOG/'summary.txt').write_text(summary+'\n');print(summary)
raise SystemExit(any(out is None for _,out,_ in results))
