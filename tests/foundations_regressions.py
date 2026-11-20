#!/usr/bin/env python3
"""Focused counterexamples for the foundations accuracy fixes; no GPU required."""
from pathlib import Path
import subprocess, tempfile, concurrent.futures, os
ROOT=Path(__file__).resolve().parents[1]
CASES={
'floating_semantics':('part2','021',r'''
int main(){
 Graph g;auto x=g.addInput("x");auto z=g.addConst(0.f,"z");g.addBinary(OpKind::Mul,x,z,"out");
 auto strict=algebraicSimplificationPass(g);
 if(!std::isnan(evaluate(strict,{{"x",std::numeric_limits<float>::infinity()}}).at("out")))return 1;
 Graph a;auto ax=a.addInput("x");auto az=a.addConst(0.f,"z");a.addBinary(OpKind::Add,ax,az,"out");
 auto as=algebraicSimplificationPass(a);
 if(std::signbit(evaluate(as,{{"x",-0.f}}).at("out")))return 2;
 if(algebraicSimplificationPass(g,true).size()>=strict.size())return 3;
 puts("strict Inf*0 remains NaN; signed-zero addition preserved; relaxed mode requires opt-in: PASS");
}'''),
'frontend_roundtrip':('part1','011',r'''
int main(){
 Graph g;g.addConst(-1.23456789e-20f,"negative");g.addConst(1.23456789f,"precise");g.addConst(-0.f,"negative_zero");
 Graph parsed;parseProgram(printGraphAsSource(g),parsed);
 if(!graphsStructurallyEqual(g,parsed))return 1;
 Graph different;different.addConst(2.f,"negative");different.addConst(1.23456789f,"precise");different.addConst(-0.f,"negative_zero");
 if(graphsStructurallyEqual(g,different))return 2;
 bool rejected=false;try{Graph bad;parseProgram("x = const(1.2.3)\n",bad);}catch(const std::exception&){rejected=true;}
 if(!rejected)return 3;
 puts("negative/scientific/precise/signed-zero constants round-trip; changed constants and malformed syntax detected: PASS");
}'''),
'shape_validation':('part1','010',r'''
int main(){Graph g;auto x=g.addInput("x");bool rejected=false;
 try{inferShapes(g,{{x.nodeId,Shape{{-1,4}}}});}catch(const std::exception&){rejected=true;}
 if(!rejected)return 1;
 puts("negative static shape rejected before propagation: PASS");}'''),
'shape_overflow':('part3','033',r'''
int main(){bool rejected=false;
 try{(void)numElements(Shape{{2147483647,2147483647,2147483647}});}catch(const std::overflow_error&){rejected=true;}
 if(!rejected)return 1;
 puts("shape product overflow rejected without signed overflow: PASS");}'''),
'loop_boundary':('part3','034',r'''
int main(){auto n=std::numeric_limits<long long>::max();auto t=tileLoop(Loop{"i",n},2);
 if(t.outerExtent!=n/2+1||actualInnerExtent(t,t.outerExtent-1)!=1)return 1;
 if(tileLoop(Loop{"i",0},4).outerExtent!=0)return 2;
 bool bad=false;try{(void)tileLoop(Loop{"i",-1},4);}catch(const std::exception&){bad=true;}
 if(!bad)return 3;
 bad=false;try{(void)actualInnerExtent(t,t.outerExtent);}catch(const std::out_of_range&){bad=true;}
 if(!bad)return 4;
 puts("LLONG_MAX ceiling division, partial tile, zero/negative extent and invalid index: PASS");}''')}
def run(item):
 name,(part,prefix,body)=item;p=next((ROOT/'docs'/part/'code').glob(prefix+'*.cpp'))
 with tempfile.TemporaryDirectory(prefix='hammer-regression-') as d:
  d=Path(d);src=d/'test.cpp';exe=d/'test'
  src.write_text('#include <limits>\n#include <cmath>\n#define main lesson_main\n#include "'+str(p)+'"\n#undef main\n'+body)
  r=subprocess.run([os.environ.get('CXX','clang++'),'-std=c++17','-O1','-fsanitize=undefined,float-cast-overflow','-fno-sanitize-recover=all',str(src),'-o',str(exe)],capture_output=True,text=True)
  if r.returncode:return name,False,r.stderr
  r=subprocess.run([str(exe)],capture_output=True,text=True)
  return name,r.returncode==0,r.stdout+r.stderr
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(run,CASES.items()))
for n,ok,out in results:print(n+': '+('PASS' if ok else 'FAIL')+'\n'+out)
raise SystemExit(any(not ok for _,ok,_ in results))
