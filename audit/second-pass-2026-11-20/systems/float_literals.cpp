#define main original_example_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part4/code/041_lowering_a_whole_fused_graph.cpp"
#undef main
int main(){
 Graph g;auto a=g.addInput("a");auto c=g.addConst(1e-7f,"c");g.addBinary(OpKind::Mul,a,c,"out");
 auto shapes=inferShapes(g,{{a.nodeId,Shape{{1}}},{c.nodeId,Shape{}}});std::map<int,long long> counts;for(auto& kv:shapes)counts[kv.first]=numElements(kv.second);
 std::map<std::string,std::vector<float>> inputs={{"a",{1e8f}}};auto expected=evaluateArrays(g,inputs,counts);
 std::string source=generateFullProgram(g,shapes,counts,inputs);
 std::ofstream f("/tmp/hammer-second-systems/float_generated.cpp");f<<source;f.close();
 std::printf("interpreter out: %.9g\n",expected.at("out")[0]);
 std::printf("serialized constant: %s\n",formatLiteralArray({1e-7f}).c_str());
}
