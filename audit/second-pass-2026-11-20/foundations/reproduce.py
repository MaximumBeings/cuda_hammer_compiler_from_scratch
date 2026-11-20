from pathlib import Path
import subprocess
ROOT=Path('/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch')
OUT=Path('/tmp/hammer-second-foundations')
cases={
'folded_temp_oob':('part0','004',None),
'fusion_cap':('part3','037',r'''
int main(){Graph g;std::vector<Value> v;for(int i=0;i<8;++i)v.push_back(g.addInput("x"+std::to_string(i)));
int name=0;while(v.size()>1){std::vector<Value> n;for(size_t i=0;i<v.size();i+=2)n.push_back(g.addBinary(OpKind::Add,v[i],v[i+1],"a"+std::to_string(name++)));v=n;}
auto f=boundedFusionPass(g,3);size_t largest=0;for(const auto& n:f.graph.nodes())largest=std::max(largest,n->fusedSteps.size());
printf("requested cap=3, largest fused body=%zu steps\n",largest);return largest>3?42:0;}
'''),
'loop_product_overflow':('part3','034',r'''
int main(){LoopNest nest{{Loop{"i",9223372036854775807LL},Loop{"j",2}}};printf("iterations=%lld\n",nest.totalIterations());}
'''),
'copied_negative_shape':('part3','033',r'''
int main(){Graph g;auto x=g.addInput("x");auto shapes=inferShapes(g,{{x.nodeId,Shape{{-2,3}}}});printf("inferShapes accepted %s\n",shapeStr(shapes.at(x.nodeId)).c_str());}
'''),
'duplicate_debug_names':('part1','011',r'''
int main(){Graph g;g.addInput("x");g.addInput("x");auto printed=printGraphAsSource(g);printf("%s",printed.c_str());try{Graph r;parseProgram(printed,r);}catch(const std::exception&e){printf("roundtrip rejected: %s\n",e.what());return 42;}return 0;}
'''),
'byte_count_overflow':('part3','026',r'''
int main(){Graph g;auto x=g.addInput("x");auto out=g.addUnary(OpKind::ReLU,x,"out");std::map<int,Shape>s{{x.nodeId,Shape{{2147483647,2147483647}}},{out.nodeId,Shape{{2147483647,2147483647}}}};printf("bytes=%lld\n",unfusedBytesMoved(g,s));}
'''),
'integer_literal_overflow':('part0','003',r'''
int main(){auto t=lex("2147483648");printf("literal=%d\n",t.front().numValue);}
''')}
for name,(part,prefix,body) in cases.items():
 source=next((ROOT/'docs'/part/'code').glob(prefix+'*.cpp'))
 p=OUT/(name+'.cpp')
 p.write_text(source.read_text() if body is None else '#define main lesson_main\n#include "'+str(source)+'"\n#undef main\n'+body)
 exe=OUT/name
 c=subprocess.run(['clang++','-std=c++17','-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all',str(p),'-o',str(exe)],capture_output=True,text=True)
 if c.returncode:(OUT/(name+'.txt')).write_text(c.stderr);print(name,'COMPILE FAIL');continue
 r=subprocess.run([str(exe)],capture_output=True,text=True)
 (OUT/(name+'.txt')).write_text('exit='+str(r.returncode)+'\n'+r.stdout+r.stderr)
 print(name,'exit',r.returncode,(r.stdout+r.stderr)[:160])
