//===- NvGpuSupport.cpp - MLIR Vector to GPU lowering support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to assist in the lowering of Vector operations
// to NvGPU dialect MMA operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToGPU/NvGpuSupport.h"
#include "mlir/Dialect/GPU/GPUDialect.h"

namespace mlir {
namespace nvgpu {
namespace NvvmMmaOperandBaseTileOperand8x128 {

namespace {

/// There are always 4 threads per [128|256|512] bit row.
constexpr int64_t kThreadsPerRow = 4;

constexpr int64_t kNumRowsPerTile = 8;

/// Returns the number of registers which compose a matrix fragment held by a
/// single thread.
int64_t inferNumRegistersPerMatrixFragment(gpu::MMAMatrixType type) {
  int64_t lineSize = inferTileWidthInBits(type.getElementType(),
                                          /*isAcc=*/type.getOperand() == "COp");
  auto shape = type.getShape();
  return (shape[0] / kNumRowsPerTile) *
         (shape[1] * type.getElementType().getIntOrFloatBitWidth()) / lineSize;
}

/// Returns the number of 8 x [128|256|512] bit tiles that compose the given
/// operand shape.
std::array<int64_t, 2> getTileShape(ArrayRef<int64_t> operandShape,
                                    Type elementType, int64_t lineSizeBits) {
  // For each 8x128bit square, a thread is responsible for one 32bit register.
  return {operandShape[0] / kNumRowsPerTile,
          (operandShape[1] * elementType.getIntOrFloatBitWidth()) /
              lineSizeBits};
}

} // namespace

int64_t inferTileWidthInBits(Type elementType, bool isAcc) {
  if (isAcc && elementType.getIntOrFloatBitWidth() == 32) {
    return 256;
  }
  if (elementType.getIntOrFloatBitWidth() == 64) {
    return isAcc ? 512 : 256;
  }
  return 128;
}

FailureOr<FragmentElementInfo> getRegisterType(gpu::MMAMatrixType type) {
  MLIRContext *ctx = type.getContext();
  bool isAccum = type.getOperand() == "COp";
  if (type.getElementType().isF16()) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(Float16Type::get(ctx), 2), 2, 32,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // f64 acc
  Type f64Ty = Float64Type::get(ctx);
  if (type.getElementType().isF64() && isAccum) {
    return FragmentElementInfo{LLVM::getFixedVectorType(f64Ty, 2), 2, 128,
                               inferNumRegistersPerMatrixFragment(type)};
  }

  // f64 operand
  if (type.getElementType().isF64() && !isAccum) {
    return FragmentElementInfo{f64Ty, 1, 64,
                               inferNumRegistersPerMatrixFragment(type)};
  }

  // int8 operand
  if (type.getElementType().isInteger(8)) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(IntegerType::get(ctx, 8), 4), 4, 32,
        inferNumRegistersPerMatrixFragment(type)};
  }
  // Integer 32bit acc operands
  if (type.getElementType().isInteger(32)) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(IntegerType::get(ctx, 32), 2), 2, 64,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // Floating point 32bit acc operands
  if (type.getElementType().isF32() && isAccum) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(Float32Type::get(ctx), 2), 2, 64,
        inferNumRegistersPerMatrixFragment(type)};
  }
  return failure();
}

static AffineMap getRegisterIndexToTileOffsetMap(OpBuilder &base,
                                                 Type elementType,
                                                 ArrayRef<int64_t> operandShape,
                                                 bool isAccumulator,
                                                 int64_t elementsPerRegister,
                                                 AffineExpr logicalValueId) {
  const int64_t lineSize = inferTileWidthInBits(elementType, isAccumulator);
  const int64_t elementsPerLine =
      lineSize / elementType.getIntOrFloatBitWidth();
  const std::array<int64_t, 2> num8x128bTiles =
      getTileShape(operandShape, elementType, lineSize);
  AffineExpr registerIdx = logicalValueId.floorDiv(elementsPerRegister);
  return AffineMap::get(
      2, 0,
      {(registerIdx % num8x128bTiles[0]) * 8,
       (registerIdx.floorDiv(num8x128bTiles[0])) * elementsPerLine},
      base.getContext());
}

FailureOr<AffineMap>
getLaneIdAndValueIdToOperandCoord(Location loc, OpBuilder &builder,
                                  gpu::MMAMatrixType fragmentType) {
  Type elementType = fragmentType.getElementType();
  ArrayRef<int64_t> operandShape = fragmentType.getShape();
  bool isAccumulator = fragmentType.getOperand() == "COp";
  FailureOr<nvgpu::NvvmMmaOperandBaseTileOperand8x128::FragmentElementInfo>
      regInfo = getRegisterType(fragmentType);
  if (failed(regInfo))
    return failure();

  const int64_t elementBitWidth =
      fragmentType.getElementType().getIntOrFloatBitWidth();
  const int64_t elementsPerRegister =
      regInfo->registerWidthBits / elementBitWidth;

  AffineExpr laneId, logicalValueIdDim;
  bindDims(builder.getContext(), laneId, logicalValueIdDim);

  // Determine what register logicalValueId corresponds to. Use that as a
  // linear index into the coordinate mapping `index -> (tile row, tile col)`.
  AffineMap registerIndexToTileCoord = getRegisterIndexToTileOffsetMap(
      builder, elementType, operandShape, isAccumulator, elementsPerRegister,
      logicalValueIdDim);

  auto makeMap = [&](ArrayRef<AffineExpr> dimExprs) -> AffineMap {
    return AffineMap::get(2, 0, dimExprs, builder.getContext());
  };

  auto tileRow = registerIndexToTileCoord.getResult(0);
  auto tileCol = registerIndexToTileCoord.getResult(1);
  return makeMap({tileRow + laneId.floorDiv(kThreadsPerRow),
                  tileCol + (laneId % kThreadsPerRow) * elementsPerRegister +
                      (logicalValueIdDim % elementsPerRegister)});
}

FailureOr<nvgpu::NvvmMmaOperandBaseTileOperand8x128::LdMatrixParams>
getLdMatrixParams(gpu::MMAMatrixType fragType, bool transpose) {
  LdMatrixParams params;
  params.fragmentType = fragType;
  if (fragType.getOperand() == "AOp" || fragType.getOperand() == "COp") {
    params.targetLayout = NVVM::MMALayout::row;
  } else {
    params.targetLayout = NVVM::MMALayout::col;
  }
  ArrayRef<int64_t> shape = fragType.getShape();
  params.contiguousDimType =
      transpose ? IteratorType::Parallel : IteratorType::Reduction;

  if (params.targetLayout == NVVM::MMALayout::row) {
    params.numTiles =
        (shape[0] / kNumRowsPerTile) *
        ((shape[1] * fragType.getElementType().getIntOrFloatBitWidth()) / 128);
  } else {
    params.numTiles =
        (shape[1] / kNumRowsPerTile) *
        ((shape[0] * fragType.getElementType().getIntOrFloatBitWidth()) / 128);
  }

  return params;
}

FailureOr<AffineMap>
getLaneIdToLdMatrixMatrixCoord(Location loc, OpBuilder &builder,
                               const LdMatrixParams &params) {
  // One thread per 128b row.
  const int64_t kNumThreadsPerTile = kNumRowsPerTile;
  const int bitsPerElement = static_cast<int>(
      params.fragmentType.getElementType().getIntOrFloatBitWidth());
  const int kElementsPer128b = (128 / bitsPerElement);
  ArrayRef<int64_t> operandShape = params.fragmentType.getShape();
  AffineExpr d0 = getAffineDimExpr(0, builder.getContext());

  auto makeMap = [&](ArrayRef<AffineExpr> dimExprs) -> AffineMap {
    return AffineMap::get(1, 0, dimExprs, builder.getContext());
  };

  // This case corresponds to row-major A|C or col-major B operands.
  if (params.contiguousDimType == IteratorType::Reduction) {
    AffineExpr row = d0 % (operandShape[0]);
    AffineExpr col = d0.floorDiv(operandShape[0]) * (kElementsPer128b);
    return makeMap({row, col});
  }

  // This case Corresponds to col-major A|C or row-major B operands. The
  // operandShape given is already pre-transposed (e.g. 8x16 = KxN).
  if (params.contiguousDimType == IteratorType::Parallel) {
    const int64_t num8x128bCols = (operandShape[0] * bitsPerElement) / 128;
    // Threads are assigned in groups of 8 first across columns, then to
    // rows. This is transpose of what `ldmatrix` expects, but when
    // `ldmatrix` gets the `.trans` qualifier, final the effect will be to
    // transpose just the blocks.
    auto groupIdx = d0.floorDiv(kNumThreadsPerTile);
    auto tileCol = (groupIdx % num8x128bCols);
    auto tileRow = groupIdx.floorDiv(num8x128bCols);
    return makeMap({tileCol * kElementsPer128b,
                    tileRow * kNumRowsPerTile + (d0 % kNumRowsPerTile)});
  }
  return failure();
}

} // namespace NvvmMmaOperandBaseTileOperand8x128

LogicalResult
PrepareContractToGPUMMASync::matchAndRewrite(vector::ContractionOp op,
                                             PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value res = op.getAcc();

  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m;
  AffineExpr n;
  AffineExpr k;
  bindDims(rewriter.getContext(), m, n, k);
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  auto iteratorTypes = op.getIteratorTypes().getValue();
  SmallVector<AffineMap, 4> maps = op.getIndexingMaps();
  if (iteratorTypes.size() != 3)
    return failure();
  if (!(isParallelIterator(iteratorTypes[0]) &&
        isParallelIterator(iteratorTypes[1]) &&
        isReductionIterator(iteratorTypes[2])))
    return failure();

  // The canonical form is "TNT" = A row-major, B col-major, C row-major.
  const auto canonicalForm = infer({{m, k}, {n, k}, {m, n}});
  if (maps == canonicalForm) {
    return failure();
  }
  if (maps == infer({{m, k}, {k, n}, {m, n}})) {
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
    std::swap(rhs, lhs);
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
    std::swap(rhs, lhs);
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
  } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
    std::swap(lhs, rhs);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
    std::swap(lhs, rhs);
  } else {
    return failure();
  }
  rewriter.replaceOpWithNewOp<vector::ContractionOp>(
      op, lhs, rhs, res, rewriter.getAffineMapArrayAttr(canonicalForm),
      op.getIteratorTypes());
  return success();
}

} // namespace nvgpu
} // namespace mlir