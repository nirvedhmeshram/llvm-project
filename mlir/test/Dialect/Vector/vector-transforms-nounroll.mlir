// RUN: mlir-opt %s -test-vector-to-vector-lowering -split-input-file| FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x16x8xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %{{.*}}[0] : vector<1x16x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R0]], %[[R1]], %[[R2]] : vector<16x8xf32>, vector<8x16xf32> into vector<16x16xf32>
//  CHECK-NEXT:   %[[R4:.+]] = vector.broadcast %[[R3]] : vector<16x16xf32> to vector<1x16x16xf32>
//  CHECK-NEXT:  return %[[R4]] : vector<1x16x16xf32>

#contraction_accesses0 = [
  affine_map<(l, i, j, k) -> (l, i, k)>,
  affine_map<(l, i, j, k) -> (l, k, j)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

func @cast_away_contraction_leading_one_dims(%arg0: vector<1x16x8xf32>, %arg1: vector<1x8x16xf32>, %arg2: vector<1x16x16xf32>) -> vector<1x16x16xf32> {
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2  : vector<1x16x8xf32>, vector<1x8x16xf32> into vector<1x16x16xf32>
  return %0: vector<1x16x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_transposeneeded
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x8xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R1]], %[[R0]], %[[R2]] : vector<8xf32>, vector<8x16xf32> into vector<16xf32>
//  CHECK-NEXT:   %[[R4:.+]] = vector.broadcast %[[R3]] : vector<16xf32> to vector<1x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] = vector.broadcast %[[R4]] : vector<1x16xf32> to vector<1x1x16xf32>
//  CHECK-NEXT:  return %[[R5]] : vector<1x1x16xf32>

#contraction_accesses1 = [
  affine_map<(l, i, j, k) -> (i, l, k)>,
  affine_map<(l, i, j, k) -> (l, k, j)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

func @cast_away_contraction_leading_one_dims_transposeneeded(%arg0: vector<1x1x8xf32>, %arg1: vector<1x8x16xf32>, %arg2: vector<1x1x16xf32>) -> vector<1x1x16xf32> {
  %0 = vector.contract #contraction_trait1 %arg0, %arg1, %arg2  : vector<1x1x8xf32>, vector<1x8x16xf32> into vector<1x1x16xf32>
  return %0: vector<1x1x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_transposeneeded2
//  CHECK-NEXT:   %[[R0:.+]] =  vector.transpose %{{.*}}[1, 0, 2] : vector<8x1x16xf32> to vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.transpose %{{.*}}[2, 0, 1] : vector<2x8x1xf32> to vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %[[R0]][0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] =  vector.extract %[[R1]][0] : vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R4:.+]] =  vector.extract %{{.*}}[0] : vector<1x2x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R2]], %[[R3]], %[[R4]] : vector<8x16xf32>, vector<2x8xf32> into vector<2x16xf32>
//  CHECK-NEXT:   %[[R6:.+]] = vector.broadcast %[[R5]] : vector<2x16xf32> to vector<1x2x16xf32>
//  CHECK-NEXT:  return %[[R6]] : vector<1x2x16xf32>

#contraction_accesses2 = [
  affine_map<(l, i, j, k) -> (k, l, j)>,
  affine_map<(l, i, j, k) -> (i, k, l)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait2 = {
  indexing_maps = #contraction_accesses2,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}


func @cast_away_contraction_leading_one_dims_transposeneeded2(%arg0: vector<8x1x16xf32>, %arg1: vector<2x8x1xf32>, %arg2: vector<1x2x16xf32>) -> vector<1x2x16xf32> {
  %0 = vector.contract #contraction_trait2 %arg0, %arg1, %arg2  : vector<8x1x16xf32>, vector<2x8x1xf32> into vector<1x2x16xf32>
  return %0: vector<1x2x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>


// CHECK-LABEL: cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x1x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0] : vector<1x2x8x1xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.transpose %[[R0]], [1, 0, 2] : vector<8x1x16xf32> to vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] =  vector.transpose %[[R1]], [2, 0, 1] : vector<2x8x1xf32> to vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R4:.+]] =  vector.extract %[[R2]][0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] =  vector.extract %[[R3]][0] : vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R6:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x2x16xf32>
//  CHECK-NEXT:   %[[R7:.+]] =  vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R4]], %[[R5]], %[[R6]] : vector<8x16xf32>, vector<2x8xf32> into vector<2x16xf32>
//  CHECK-NEXT:   %[[R8:.+]] =  vector.broadcast %[[R7]] : vector<2x16xf32> to vector<1x2x16xf32>
//  CHECK-NEXT:   %[[R9:.+]] =  vector.broadcast %[[R8]] : vector<1x2x16xf32> to vector<1x1x2x16xf32>
//  CHECK-NEXT:  return %[[R9]] : vector<1x1x2x16xf32>

#contraction_accesses2 = [
  affine_map<(m, l, i, j, k) -> (m, k, l, j)>,
  affine_map<(m, l, i, j, k) -> (m, i, k, l)>,
  affine_map<(m, l, i, j, k) -> (m, l, i, j)>
]
#contraction_trait2 = {
  indexing_maps = #contraction_accesses2,
  iterator_types = ["parallel","parallel", "parallel", "parallel", "reduction"]
}


func @cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4(%arg0: vector<1x8x1x16xf32>, %arg1: vector<1x2x8x1xf32>, %arg2: vector<1x1x2x16xf32>) -> vector<1x1x2x16xf32> {
  %0 = vector.contract #contraction_trait2 %arg0, %arg1, %arg2  : vector<1x8x1x16xf32>, vector<1x2x8x1xf32> into vector<1x1x2x16xf32>
  return %0: vector<1x1x2x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d3)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4_negativecase
//  CHECK-NEXT:   %[[R0:.+]] =  vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %{{.*}}, %{{.*}}, %{{.*}} : vector<1x8x1x16xf32>, vector<1x2x8x1xf32> into vector<1x1x2x16xf32>
//  CHECK-NEXT:  return %[[R0]] : vector<1x1x2x16xf32>

#contraction_accesses3 = [
  affine_map<(m, l, i, j, k) -> (m, k, l, j)>,
  affine_map<(m, l, i, j, k) -> (m, i, k, l)>,
  affine_map<(m, l, i, j, k) -> (l, m, i, j)>
]
#contraction_trait3 = {
  indexing_maps = #contraction_accesses3,
  iterator_types = ["parallel","parallel", "parallel", "parallel", "reduction"]
}

func @cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4_negativecase(%arg0: vector<1x8x1x16xf32>, %arg1: vector<1x2x8x1xf32>, %arg2: vector<1x1x2x16xf32>) -> vector<1x1x2x16xf32> {
  %0 = vector.contract #contraction_trait3 %arg0, %arg1, %arg2  : vector<1x8x1x16xf32>, vector<1x2x8x1xf32> into vector<1x1x2x16xf32>
  return %0: vector<1x1x2x16xf32>
}