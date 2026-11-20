#define main original_example_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part5/code/057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp"
#undef main
#include <cstdlib>
int main() {
 std::string log;
 const std::string path="/tmp/hammer-second-systems/generated";
 bool first=compileToSharedLibrary("extern \"C\" int value() { return 37; }",path+".cpp",path+".so",log);
 if(!first) return 2;
 bool normalFailed=compileToSharedLibrary("invalid source",path+".cpp",path+".so",log);
 { JitModule stale(path+".so");auto old=stale.getFunction<int(*)()>("value");
 std::printf("normal compile error; helper success=%s; following caller's unconditional dlopen still runs stale function=%d\n",normalFailed?"true":"false",old()); }
 std::string oldPath=std::getenv("PATH");
 std::string newPath="/tmp/hammer-second-systems/bin:"+oldPath;
 setenv("PATH",newPath.c_str(),1);
 bool failedCompileClaimedSuccess=compileToSharedLibrary("invalid source",path+".cpp",path+".so",log);
 JitModule module(path+".so");auto fn=module.getFunction<int(*)()>("value");
 std::printf("compiler exit1, empty stderr; helper returned success=%s; stale function=%d\n",failedCompileClaimedSuccess?"true":"false",fn());
}
