#define main lesson_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp"
#undef main

int main(){Graph g;auto x=g.addInput("x");auto out=g.addUnary(OpKind::ReLU,x,"out");std::map<int,Shape>s{{x.nodeId,Shape{{2147483647,2147483647}}},{out.nodeId,Shape{{2147483647,2147483647}}}};printf("bytes=%lld\n",unfusedBytesMoved(g,s));}
