#define main lesson_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp"
#undef main

int main(){Graph g;std::vector<Value> v;for(int i=0;i<8;++i)v.push_back(g.addInput("x"+std::to_string(i)));
int name=0;while(v.size()>1){std::vector<Value> n;for(size_t i=0;i<v.size();i+=2)n.push_back(g.addBinary(OpKind::Add,v[i],v[i+1],"a"+std::to_string(name++)));v=n;}
auto f=boundedFusionPass(g,3);size_t largest=0;for(const auto& n:f.graph.nodes())largest=std::max(largest,n->fusedSteps.size());
printf("requested cap=3, largest fused body=%zu steps\n",largest);return largest>3?42:0;}
