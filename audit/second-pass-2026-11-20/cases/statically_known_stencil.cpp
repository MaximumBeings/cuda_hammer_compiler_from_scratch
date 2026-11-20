#define main book_original_main
#include "/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part7/code/091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp"
#undef main
int main() {
  Graph g;
  Value a=g.addInput("left"), b=g.addInput("center"), c=g.addInput("right");
  Value quarter=g.addConst(.25f,"quarter"), half=g.addConst(.5f,"half");
  Value x=g.addBinary(OpKind::Mul,a,quarter,"x"), y=g.addBinary(OpKind::Mul,b,half,"y"), z=g.addBinary(OpKind::Mul,c,quarter,"z");
  Value xy=g.addBinary(OpKind::Add,x,y,"xy");
  g.addBinary(OpKind::Add,xy,z,"out");
  std::map<int,long long> counts;for(const auto& n:g.nodes()) counts[n->id]=1;
  // Graph edges are already complete before any input data below exists.
  auto result=evaluateArrays(g,{{"left",{4.f}},{"center",{8.f}},{"right",{12.f}}},counts);
  printf("Static scalar stencil: %.1f; expected 8.0; no shift/gather primitive or im2col staging\n",result.at("out").at(0));
  return result.at("out").at(0)==8.f ? 0 : 1;
}
