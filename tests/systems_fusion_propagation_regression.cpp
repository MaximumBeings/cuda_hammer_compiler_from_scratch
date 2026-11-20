// Verify later codegen lessons retain the corrected foundation contracts.
#define main chapter17_example_main
#include "../docs/part4/code/041_lowering_a_whole_fused_graph.cpp"
#undef main
int main(){
 Graph g;std::vector<Value> level;std::map<int,Shape> declared;std::map<std::string,std::vector<float>> inputs;
 for(int i=0;i<8;++i){std::string name="x"+std::to_string(i);auto value=g.addInput(name);level.push_back(value);declared[value.nodeId]=Shape{};inputs[name]={float(i+1)};}
 int serial=0;while(level.size()>1){std::vector<Value> next;for(size_t i=0;i<level.size();i+=2)next.push_back(g.addBinary(OpKind::Add,level[i],level[i+1],"n"+std::to_string(serial++)));level=next;}
 auto shapes=inferShapes(g,declared);std::map<int,long long> counts;for(auto& kv:shapes)counts[kv.first]=numElements(kv.second);
 auto result=boundedReductionFusionPass(g,counts,3);std::map<int,long long> fusedCounts;
 for(auto& n:result.graph.nodes()) {if(n->fusedSteps.size()>3)return 1;fusedCounts[n->id]=counts.at(result.representativeOldId.at(n->id));}
 auto expected=evaluateArrays(g,inputs,counts);auto actual=evaluateArrays(result.graph,inputs,fusedCounts);
 if(actual.at("n6")[0]!=expected.at("n6")[0] || actual.at("n6")[0]!=36)return 2;
 bool invalid=false;declared[g.nodes()[0]->id]=Shape{{0}};try{inferShapes(g,declared);}catch(const std::exception&){invalid=true;}if(!invalid)return 3;
 bool overflow=false;try{numElements(Shape{{2147483647,2147483647,2147483647}});}catch(const std::overflow_error&){overflow=true;}if(!overflow)return 4;
 std::puts("balanced-tree fusion bodies obey3-step cap and preserve result36");
 std::puts("later backend rejects invalid leaf shapes and count overflow before allocation/codegen");
}
