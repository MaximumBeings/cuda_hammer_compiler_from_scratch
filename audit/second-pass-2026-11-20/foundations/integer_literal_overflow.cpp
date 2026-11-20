#define main lesson_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part0/code/003_parsing_lowering_and_ir.cpp"
#undef main

int main(){auto t=lex("2147483648");printf("literal=%d\n",t.front().numValue);}
