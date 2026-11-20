#define main lesson_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part1/code/011_printing_the_ir_as_reparseable_text.cpp"
#undef main

int main(){Graph g;g.addInput("x");g.addInput("x");auto printed=printGraphAsSource(g);printf("%s",printed.c_str());try{Graph r;parseProgram(printed,r);}catch(const std::exception&e){printf("roundtrip rejected: %s\n",e.what());return 42;}return 0;}
