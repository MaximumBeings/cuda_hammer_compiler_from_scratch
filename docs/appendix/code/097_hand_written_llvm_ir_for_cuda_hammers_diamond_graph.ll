; Appendix C.1 -- CUDA Hammer's own Chapter 4 diamond graph, hand-written
; directly as real LLVM IR text instead of built through Graph::addInput/
; addBinary/addUnary. The same five nodes File 005 built by hand:
;
;   a, b = Input, Input
;   t1   = Add(a, b)
;   t2   = Mul(t1, a)
;   t3   = ReLU(t1)
;   out  = Add(t2, t3)
;
; with a = 3.0, b = 4.0, giving t1=7, t2=21, t3=7, out=28 by hand.
;
; ReLU(t1) is expressed the only way LLVM IR has to express a
; conditional-select over an SSA value: a real fcmp (compare t1 to 0.0)
; feeding a real select. This is not a simplification chosen for this
; book -- it is the same fcmp+select shape Chapter 34's own
; max(a,b)=b+ReLU(a-b) identity already used at the C++ source level;
; here it appears one level lower, in the compiler's own IR.

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
