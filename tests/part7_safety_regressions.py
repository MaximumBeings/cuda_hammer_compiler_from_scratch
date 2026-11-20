#!/usr/bin/env python3
"""Compile standalone Part7 copies and test shape/count/body-budget rejection."""
from pathlib import Path
import json, subprocess, tempfile
ROOT=Path(__file__).resolve().parents[1]
base=r'''
int main() {
    int passed=0;
    Graph g; Value x=g.addInput("x");
    for (int d : {0,-2}) {
        bool rejected=false;
        try { (void)inferShapes(g, {{x.nodeId, Shape{{d}}}}); }
        catch (const std::exception&) { rejected=true; }
        if (!rejected) return 1;
        ++passed;
    }
    bool rejected=false;
    try { (void)numElements(Shape{{2147483647,2147483647,2147483647}}); }
    catch (const std::overflow_error&) { rejected=true; }
    if (!rejected) return 2;
    ++passed;
    auto valid=inferShapes(g, {{x.nodeId, Shape{{1}}}});
    if (numElements(valid.at(x.nodeId)) != 1) return 3;
    ++passed;
    bool addRejected=false,mulRejected=false;
    try { (void)checkedCountAdd(std::numeric_limits<long long>::max(),1); }
    catch (const std::overflow_error&) { addRejected=true; }
    try { (void)checkedCountMul(std::numeric_limits<long long>::max(),2); }
    catch (const std::overflow_error&) { mulRejected=true; }
    if (!addRejected || !mulRejected) return 4;
    passed+=2;
    Graph badBroadcast;
    Value lhsInput=badBroadcast.addInput("lhs"), rhsInput=badBroadcast.addInput("rhs");
    badBroadcast.addBinary(OpKind::Add,lhsInput,rhsInput,"sum");
    bool broadcastRejected=false;
    try { (void)inferShapes(badBroadcast, {{lhsInput.nodeId,Shape{{2}}},{rhsInput.nodeId,Shape{{3}}}}); }
    catch(const std::exception&) { broadcastRejected=true; }
    if(!broadcastRejected) return 7;
    ++passed;
    BODY_BUDGET_CHECK
    printf("PASS %d Part7 safety checks\n",passed);
    return 0;
}
'''
cap=r'''
    Graph tree;
    auto a=tree.addInput("a"),b=tree.addInput("b"),c=tree.addInput("c"),d=tree.addInput("d");
    auto l=tree.addUnary(OpKind::ReLU,tree.addBinary(OpKind::Add,a,b,"left"),"lrelu");
    auto r=tree.addUnary(OpKind::ReLU,tree.addBinary(OpKind::Add,c,d,"right"),"rrelu");
    tree.addBinary(OpKind::Add,l,r,"out");
    std::map<int,long long> counts; for(const auto& n:tree.nodes()) counts[n->id]=1;
    auto fused=boundedReductionFusionPass(tree,counts,3);
    for(const auto& n:fused.graph.nodes()) if(n->fusedSteps.size()>3) return 5;
    ++passed;
    bool capRejected=false;
    try { (void)boundedReductionFusionPass(tree,counts,0); }
    catch(const std::invalid_argument&) { capRejected=true; }
    if(!capRejected) return 6;
    ++passed;
'''
with tempfile.TemporaryDirectory(prefix='hammer-part7-safety-') as temp:
    td=Path(temp)
    for p in sorted((ROOT/'docs/part7/code').glob('*.cpp')):
        src='#define main book_example_main\n#include '+json.dumps(str(p))+'\n#undef main\n'
        src+=base.replace('BODY_BUDGET_CHECK',cap if 'boundedReductionFusionPass(' in p.read_text() else '')
        cpp=td/'check.cpp';cpp.write_text(src);exe=td/'check'
        subprocess.run(['clang++','-std=c++17','-Wall','-Wextra','-O2',str(cpp),'-o',str(exe)],check=True)
        run=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
        print(p.name+': '+run.stdout.strip(),flush=True)
print('ALL 12 standalone Part7 copies passed shape/count regressions; four also passed body-budget regressions')
