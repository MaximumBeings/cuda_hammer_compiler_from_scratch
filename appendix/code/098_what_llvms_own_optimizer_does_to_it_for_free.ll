; Appendix C.2 -- the same diamond graph as File 097, but with a and b
; baked in as literal constants inside main, the LLVM IR analogue of
; Chapter 9's constantFoldPass() input: every operand of every node is
; already a known constant before optimization runs.
;
; Chapter 9 hand-wrote constantFoldPass() and deadCodeEliminationPass()
; specifically for CUDA Hammer's own OpKind enum and Graph/Node types.
; This file asks what a REAL, general-purpose, not-tensor-specific
; optimizer pipeline -- one that has never heard of Value, Node, Graph,
; or OpKind -- does to the exact same hand-translated computation.

@.fmt = private unnamed_addr constant [30 x i8] c"LLVM IR diamond result: %.6f\0A\00"

declare i32 @printf(ptr, ...)

define float @diamond(float %a, float %b) {
entry:
  %t1 = fadd float %a, %b
  %t2 = fmul float %t1, %a
  %t3cmp = fcmp ogt float %t1, 0.000000e+00
  %t3 = select i1 %t3cmp, float %t1, float 0.000000e+00
  %out = fadd float %t2, %t3
  ret float %out
}

define i32 @main() {
entry:
  %r = call float @diamond(float 3.000000e+00, float 4.000000e+00)
  %rd = fpext float %r to double
  %ignore = call i32 (ptr, ...) @printf(ptr @.fmt, double %rd)
  ret i32 0
}
