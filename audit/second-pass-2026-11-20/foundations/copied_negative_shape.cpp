#define main lesson_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp"
#undef main

int main(){Graph g;auto x=g.addInput("x");auto shapes=inferShapes(g,{{x.nodeId,Shape{{-2,3}}}});printf("inferShapes accepted %s\n",shapeStr(shapes.at(x.nodeId)).c_str());}
