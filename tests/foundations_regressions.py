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
# Second-pass defects: the existing example is itself a memory-safety regression.
CASES.update({
'folded_original_driver':('part0','004',r"""
int main(){return lesson_main();}
"""),
'integer_domain':('part0','003',r"""
int main(){int rejected=0;
 auto check=[&](auto f){try{f();}catch(const std::exception&){++rejected;}};
 check([]{(void)lex("2147483648");});
 check([]{(void)applyOp('+',std::numeric_limits<int>::max(),1);});
 check([]{(void)applyOp('*',50000,50000);});
 check([]{(void)applyOp('/',1,0);});
 check([]{(void)applyOp('/',std::numeric_limits<int>::min(),-1);});
 if(rejected!=5||applyOp('+',40,2)!=42)return 1;
 puts("out-of-range literals and arithmetic diagnosed; valid integer arithmetic retained: PASS");}
"""),
'branching_fusion_budget':('part3','037',r"""
int main(){Graph g;std::vector<Value> layer;std::map<std::string,float> input;
 for(int i=0;i<8;++i){auto name="x"+std::to_string(i);layer.push_back(g.addInput(name));input[name]=float(i+1);}
 int id=0;while(layer.size()>1){std::vector<Value> next;for(size_t i=0;i<layer.size();i+=2)
 next.push_back(g.addBinary(OpKind::Add,layer[i],layer[i+1],"n"+std::to_string(id++)));layer=next;}
 for(int cap:{1,2,3,4,8}){auto f=boundedFusionPass(g,cap);
 for(const auto& n:f.graph.nodes())if(n->fusedSteps.size()>size_t(cap))return 1;
 if(evaluate(f.graph,input).at("n6")!=evaluate(g,input).at("n6"))return 2;}
 bool rejected=false;try{(void)boundedFusionPass(g,0);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected)return 3;
 puts("balanced branching tree obeys body budgets1/2/3/4/8 with matching results; cap0 rejected: PASS");}
"""),
'copied_shape_inference':('part3','033',r"""
int main(){Graph g;auto x=g.addInput("x");int rejected=0;
 for(auto shape:{Shape{{-2,3}},Shape{{0,3}},Shape{{2147483647,2147483647,2147483647}}})
 try{(void)inferShapes(g,{{x.nodeId,shape}});}catch(const std::exception&){++rejected;}
 if(rejected!=3)return 1;
 puts("copied inference rejects negative/zero dimensions and unrepresentable products: PASS");}
"""),
'aggregate_overflow':('part3','026',r"""
int main(){Graph g;auto x=g.addInput("x");auto out=g.addUnary(OpKind::ReLU,x,"out");
 std::map<int,Shape>s{{x.nodeId,Shape{{2147483647,2147483647}}},{out.nodeId,Shape{{2147483647,2147483647}}}};
 bool bad=false;try{(void)unfusedBytesMoved(g,s);}catch(const std::overflow_error&){bad=true;}
 if(!bad)return 1;
 bad=false;try{(void)checkedCountAdd(std::numeric_limits<long long>::max(),1);}catch(const std::overflow_error&){bad=true;}
 if(!bad)return 2;
 puts("representable shapes with oversized traffic and aggregate count overflow rejected: PASS");}
"""),
'loop_total_overflow':('part3','034',r"""
int main(){LoopNest n{{Loop{"i",std::numeric_limits<long long>::max()},Loop{"j",2}}};
 bool bad=false;try{(void)n.totalIterations();}catch(const std::overflow_error&){bad=true;}
 if(!bad)return 1;
 puts("loop iteration product overflow rejected: PASS");}
""")})

# Exercise each standalone inference copy, not only Chapter 6's canonical helper.
for _prefix in ['009','010','026','029','030','031','032','035','036','038']:
    CASES['shape_copy_'+_prefix] = ('part1' if _prefix in ['009','010'] else 'part3', _prefix,
        CASES['copied_shape_inference'][2])

def run(item):
 name,(part,prefix,body)=item;p=next((ROOT/'docs'/part/'code').glob(prefix+'*.cpp'))
 with tempfile.TemporaryDirectory(prefix='hammer-regression-') as d:
  d=Path(d);src=d/'test.cpp';exe=d/'test'
  src.write_text('#include <limits>\n#include <cmath>\n#define main lesson_main\n#include "'+str(p)+'"\n#undef main\n'+body)
  r=subprocess.run([os.environ.get('CXX','clang++'),'-std=c++17','-O1','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all',str(src),'-o',str(exe)],capture_output=True,text=True)
  if r.returncode:return name,False,r.stderr
  r=subprocess.run([str(exe)],capture_output=True,text=True)
  return name,r.returncode==0,r.stdout+r.stderr
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(run,CASES.items()))
for n,ok,out in results:print(n+': '+('PASS' if ok else 'FAIL')+'\n'+out)
raise SystemExit(any(not ok for _,ok,_ in results))
