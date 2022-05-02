//===- NvvmMMASupport.h - MLIR Vector to GPU lowering support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to assist in the lowering of Vector operations
// to GPU dialect MMA operations.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H
#define MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace nvgpu {

/// Helps to calculate the offsets within the tile for any NVVM/PTX MMA operand
/// that has a base tile size of 8 elements x [128|256|512] bits
namespace NvvmMmaOperandBaseTileOperand8x128 {

/// Returns the number of bits in a single tile row. It is either 128, 256, or
/// 512 bits depending on the data type and whether the operand is an
/// accumulator.
int64_t inferTileWidthInBits(Type elementType, bool isAcc);

/// Specifies information about the registers which compose a matrix fragment
/// according to the PTX documentation.
struct FragmentElementInfo {
  Type registerLLVMType;
  int64_t elementsPerRegister;
  int64_t registerWidthBits;
  int64_t numRegistersPerFragment;
};

/// Returns a FragmentElementInfo struct describing the register types for the
/// given matrix fragment type.
FailureOr<FragmentElementInfo> getRegisterType(gpu::MMAMatrixType type);

/// Returns an AffineMap which maps a two dimensions representing (laneId,
/// logicalValueId) and returns two results representing offsets within a
/// matrix operand. The offsets point to the values the thread is responsible
/// for (AKA the matrix fragment values) during a warp-collective matrix
/// operation. For a visual reference of this LaneId -> (row, col) mapping,
/// please see NVIDIA's PTX documentation:
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
FailureOr<AffineMap>
getLaneIdAndValueIdToOperandCoord(Location loc, OpBuilder &builder,
                                  gpu::MMAMatrixType fragmentType);

struct LdMatrixParams {
  gpu::MMAMatrixType fragmentType;
  int64_t numTiles;
  IteratorType contiguousDimType;
  NVVM::MMALayout targetLayout;
};

FailureOr<LdMatrixParams> getLdMatrixParams(gpu::MMAMatrixType fragType,
                                            bool transpose);
/// Returns an AffineMap which maps a single dimension representing the laneId
/// to two results representing offsets within the matrix operand that should
/// be the pointer locations a thread should pass to the ldmatrix instruction.
FailureOr<AffineMap>
getLaneIdToLdMatrixMatrixCoord(Location loc, OpBuilder &builder,
                               const LdMatrixParams &params);

} // namespace NvvmMmaOperandBaseTileOperand8x128

// Transform contract into (m, k)x(n, k)x(m, n) form so that it can be converted
// to MMA matmul.
struct PrepareContractToGPUMMASync
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace nvgpu
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H
