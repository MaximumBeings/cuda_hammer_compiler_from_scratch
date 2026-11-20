// Host-only validation: exercises the actual CUDA text emitter, not a GPU.
#define main chapter18_example_main
#include "../docs/part4/code/044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu"
#undef main
int main() {
 Graph g;auto a=g.addInput("a");auto c=g.addConst(1e-7f,"c");g.addBinary(OpKind::Mul,a,c,"y");
 auto shapes=inferShapes(g,{{a.nodeId,Shape{{1}}},{c.nodeId,Shape{}}});std::map<int,long long> counts;for(auto& kv:shapes)counts[kv.first]=numElements(kv.second);
 auto source=generateFullCudaProgram(g,shapes,counts,{{"a",{1e8f}}});
 if(source.find("float h_c[] = {"+floatLiteral(1e-7f)+"};")==std::string::npos)return 1;
 if(source.find("kernel result y: %.9g")==std::string::npos)return 2;
 if(source.find("return 77;")==std::string::npos)return 3;
 bool rejected=false;try{runShellCaptureAll("exit 77");}catch(const std::runtime_error&){rejected=true;}if(!rejected)return 4;
 if(runShellCaptureAll("printf diagnostic; exit 77",true)!="diagnostic")return 5;
 std::puts("CUDA text retains exact small constant, round-trip stdout and fail-fast diagnostics");
 std::puts("compiler exit77 rejected; only explicit CUDA diagnostic execution may return77");
 std::puts("host-only test: generated CUDA device code was not compiled or executed");
}
