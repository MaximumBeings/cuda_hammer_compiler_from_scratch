// Appendix C.3 -- the same diamond graph (t1=Add(a,b), t2=Mul(t1,a),
// t3=ReLU(t1), out=Add(t2,t3)) applied elementwise over a 5-element
// array, expressed as ONE real MLIR linalg.generic op -- the declarative,
// fusion-friendly form Chapter 15's own hand-written LoopNest::tileLoop()
// and loopNestsCompatibleForFusion() exist to approximate by hand for
// CUDA Hammer's own IR.
//
// Inputs (chosen so ReLU genuinely clips something away, unlike a
// convenience all-positive test vector):
//   a = [ 1, -2,   3, -4,  5]
//   b = [10,  1, -30,  1, -3]
// giving, by hand, per index i:
//   i=0: t1=11  (pos) t2=11   t3=11  out=22
//   i=1: t1=-1  (neg) t2=2    t3=0   out=2
//   i=2: t1=-27 (neg) t2=-81  t3=0   out=-81
//   i=3: t1=-3  (neg) t2=12   t3=0   out=12
//   i=4: t1=2   (pos) t2=10   t3=2   out=12
//
// main() returns out[3] = 12 -- the index where ReLU clips a nonzero
// t1 away entirely (t3 goes from what would be -3 to 0).
#map = affine_map<(i) -> (i)>

func.func @diamond_elementwise(%a: memref<5xf32>, %b: memref<5xf32>, %out: memref<5xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%a, %b : memref<5xf32>, memref<5xf32>) outs(%out : memref<5xf32>) {
  ^bb0(%av: f32, %bv: f32, %outv: f32):
    %t1 = arith.addf %av, %bv : f32
    %t2 = arith.mulf %t1, %av : f32
    %zero = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %t1, %zero : f32
    %t3 = arith.select %cmp, %t1, %zero : f32
    %o = arith.addf %t2, %t3 : f32
    linalg.yield %o : f32
  }
  return
}

func.func @main() -> f32 {
  %a = memref.alloc() : memref<5xf32>
  %b = memref.alloc() : memref<5xf32>
  %out = memref.alloc() : memref<5xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %a0 = arith.constant 1.0 : f32
  %a1 = arith.constant -2.0 : f32
  %a2 = arith.constant 3.0 : f32
  %a3 = arith.constant -4.0 : f32
  %a4 = arith.constant 5.0 : f32
  memref.store %a0, %a[%c0] : memref<5xf32>
  memref.store %a1, %a[%c1] : memref<5xf32>
  memref.store %a2, %a[%c2] : memref<5xf32>
  memref.store %a3, %a[%c3] : memref<5xf32>
  memref.store %a4, %a[%c4] : memref<5xf32>

  %b0 = arith.constant 10.0 : f32
  %b1 = arith.constant 1.0 : f32
  %b2 = arith.constant -30.0 : f32
  %b3 = arith.constant 1.0 : f32
  %b4 = arith.constant -3.0 : f32
  memref.store %b0, %b[%c0] : memref<5xf32>
  memref.store %b1, %b[%c1] : memref<5xf32>
  memref.store %b2, %b[%c2] : memref<5xf32>
  memref.store %b3, %b[%c3] : memref<5xf32>
  memref.store %b4, %b[%c4] : memref<5xf32>

  call @diamond_elementwise(%a, %b, %out) : (memref<5xf32>, memref<5xf32>, memref<5xf32>) -> ()

  %r = memref.load %out[%c3] : memref<5xf32>
  return %r : f32
}
