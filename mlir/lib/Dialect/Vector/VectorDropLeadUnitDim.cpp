//===- VectorDropLeadUnitDim.cpp - Conversion within the Vector dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-drop-unit-dim"

using namespace mlir;
using namespace mlir::vector;

// Trims leading one dimensions from `oldType` and returns the result type.
// Returns `vector<1xT>` if `oldType` only has one element.
static VectorType trimLeadingOneDims(VectorType oldType) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape =
      oldShape.drop_while([](int64_t dim) { return dim == 1; });
  // Make sure we have at least 1 dimension per vector type requirements.
  if (newShape.empty())
    newShape = oldShape.take_back();
  return VectorType::get(newShape, oldType.getElementType());
}

/// Return a smallVector of size `rank` containing all zeros.
static SmallVector<int64_t> splatZero(int64_t rank) {
  return SmallVector<int64_t>(rank, 0);
}
namespace {

// Casts away leading one dimensions in vector.extract_strided_slice's vector
// input by inserting vector.shape_cast.
struct CastAwayExtractStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    // vector.extract_strided_slice requires the input and output vector to have
    // the same rank. Here we drop leading one dimensions from the input vector
    // type to make sure we don't cause mismatch.
    VectorType oldSrcType = extractOp.getVectorType();
    VectorType newSrcType = trimLeadingOneDims(oldSrcType);

    if (newSrcType.getRank() == oldSrcType.getRank())
      return failure();

    int64_t dropCount = oldSrcType.getRank() - newSrcType.getRank();

    VectorType oldDstType = extractOp.getType();
    VectorType newDstType =
        VectorType::get(oldDstType.getShape().drop_front(dropCount),
                        oldDstType.getElementType());

    Location loc = extractOp.getLoc();

    Value newSrcVector = rewriter.create<vector::ExtractOp>(
        loc, extractOp.vector(), splatZero(dropCount));

    // The offsets/sizes/strides attribute can have a less number of elements
    // than the input vector's rank: it is meant for the leading dimensions.
    auto newOffsets = rewriter.getArrayAttr(
        extractOp.offsets().getValue().drop_front(dropCount));
    auto newSizes = rewriter.getArrayAttr(
        extractOp.sizes().getValue().drop_front(dropCount));
    auto newStrides = rewriter.getArrayAttr(
        extractOp.strides().getValue().drop_front(dropCount));

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, newDstType, newSrcVector, newOffsets, newSizes, newStrides);

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(extractOp, oldDstType,
                                                     newExtractOp);

    return success();
  }
};

// Casts away leading one dimensions in vector.extract_strided_slice's vector
// inputs by inserting vector.shape_cast.
struct CastAwayInsertStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldSrcType = insertOp.getSourceVectorType();
    VectorType newSrcType = trimLeadingOneDims(oldSrcType);
    VectorType oldDstType = insertOp.getDestVectorType();
    VectorType newDstType = trimLeadingOneDims(oldDstType);

    int64_t srcDropCount = oldSrcType.getRank() - newSrcType.getRank();
    int64_t dstDropCount = oldDstType.getRank() - newDstType.getRank();
    if (srcDropCount == 0 && dstDropCount == 0)
      return failure();

    // Trim leading one dimensions from both operands.
    Location loc = insertOp.getLoc();

    Value newSrcVector = rewriter.create<vector::ExtractOp>(
        loc, insertOp.source(), splatZero(srcDropCount));
    Value newDstVector = rewriter.create<vector::ExtractOp>(
        loc, insertOp.dest(), splatZero(dstDropCount));

    auto newOffsets = rewriter.getArrayAttr(
        insertOp.offsets().getValue().take_back(newDstType.getRank()));
    auto newStrides = rewriter.getArrayAttr(
        insertOp.strides().getValue().take_back(newSrcType.getRank()));

    auto newInsertOp = rewriter.create<vector::InsertStridedSliceOp>(
        loc, newDstType, newSrcVector, newDstVector, newOffsets, newStrides);

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(insertOp, oldDstType,
                                                     newInsertOp);

    return success();
  }
};

// Turns vector.transfer_read on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_read on vector without leading
// 1 dimensions.
struct CastAwayTransferReadLeadingOneDim
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (read.getTransferRank() == 0)
      return failure();

    if (read.mask())
      return failure();

    auto shapedType = read.source().getType().cast<ShapedType>();
    if (shapedType.getElementType() != read.getVectorType().getElementType())
      return failure();

    VectorType oldType = read.getVectorType();
    VectorType newType = trimLeadingOneDims(oldType);

    if (newType == oldType)
      return failure();

    AffineMap oldMap = read.permutation_map();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBoundsAttr;
    if (read.in_bounds())
      inBoundsAttr = rewriter.getArrayAttr(
          read.in_boundsAttr().getValue().take_back(newType.getRank()));

    auto newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), newType, read.source(), read.indices(),
        AffineMapAttr::get(newMap), read.padding(), /*mask=*/Value(),
        inBoundsAttr);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(read, oldType, newRead);

    return success();
  }
};

// Turns vector.transfer_write on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_write on vector without leading
// 1 dimensions.
struct CastAwayTransferWriteLeadingOneDim
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (write.getTransferRank() == 0)
      return failure();

    if (write.mask())
      return failure();

    auto shapedType = write.source().getType().dyn_cast<ShapedType>();
    if (shapedType.getElementType() != write.getVectorType().getElementType())
      return failure();

    VectorType oldType = write.getVectorType();
    VectorType newType = trimLeadingOneDims(oldType);
    if (newType == oldType)
      return failure();
    int64_t dropDim = oldType.getRank() - newType.getRank();

    AffineMap oldMap = write.permutation_map();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBoundsAttr;
    if (write.in_bounds())
      inBoundsAttr = rewriter.getArrayAttr(
          write.in_boundsAttr().getValue().take_back(newType.getRank()));

    auto newVector = rewriter.create<vector::ExtractOp>(
        write.getLoc(), write.vector(), splatZero(dropDim));
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        write, newVector, write.source(), write.indices(),
        AffineMapAttr::get(newMap), inBoundsAttr);

    return success();
  }
};

// Turns vector.contract on vector with leading 1 dimensions into
// vector.extract followed by vector.contract on vector without leading
// 1 dimensions. Also performs tranpose of lhs and rhs operands if required
// prior to extract
struct CastAwayContractionLeadingOneDim
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldLhsType = contractOp.getLhsType();
    VectorType oldRhsType = contractOp.getRhsType();
    VectorType oldAccType =
        contractOp.getAccType().dyn_cast_or_null<VectorType>();
    if (oldAccType == nullptr)
      return failure();
    if (oldLhsType.getRank() < 2 ||
        oldRhsType.getRank() <
            2 || // dropping dim for rank==1 doesnt make sense
        oldAccType.getRank() < 2)
      return failure();
    if (llvm::size(contractOp.masks()) != 0)
      return failure();
    VectorType newAccType = trimLeadingOneDims(oldAccType);
    if (newAccType == oldAccType)
      return failure();
    int64_t dropDim = 1; // currently we support only dropping one dim but the
                         // pattern can be applied greedily to drop more

    auto oldIndexingMaps = contractOp.getIndexingMaps();
    SmallVector<AffineMap, 4> newIndexingMaps;

    auto oldIteratorTypes = contractOp.iterator_types();
    SmallVector<Attribute, 4> newIteratorTypes;
    std::array<bool, 3> validExtract = {
        false, false}; // to check if the dim to be dropped exists in lhs and
                       // rhs and is a leading dim

    for (const auto &it : llvm::enumerate(oldIteratorTypes)) {
      int64_t idx = it.index();
      if (idx == 0) {
        if (!isParallelIterator(
                it.value())) // reduction type iterators cannot be dropped
          return failure();
        continue;
      }
      newIteratorTypes.push_back(it.value());
    }

    auto lhs = contractOp.lhs();
    auto rhs = contractOp.rhs();
    for (const auto &it : llvm::enumerate(oldIndexingMaps)) {
      SmallVector<AffineExpr, 4> results;
      int64_t idx = it.value().getDimPosition(0);
      auto map = it.value();
      if (idx != 0) {
        if (it.index() == 0) { // lhs operand requires transpose in order to
                               // make the unit dim leading
          SmallVector<int64_t> perm;
          SmallVector<AffineExpr, 4> transposeResults;
          for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
            int64_t idx2 = map.getDimPosition(i);
            if (idx2 == 0) {
              perm.insert(perm.begin(), i);
              auto targetExpr = rewriter.getAffineDimExpr(idx2);
              transposeResults.insert(transposeResults.begin(), targetExpr);
            } else {
              perm.push_back(i);
              auto targetExpr = rewriter.getAffineDimExpr(idx2);
              transposeResults.push_back(targetExpr);
            }
          }
          map = AffineMap::get(map.getNumDims(), 0, transposeResults,
                               contractOp.getContext());
          lhs = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), lhs,
                                                     perm);
        }
        if (it.index() == 1) { // rhs operand requires transpose in order to
                               // make the unit dim leading
          SmallVector<int64_t> perm;
          SmallVector<AffineExpr, 4> transposeResults;
          for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
            int64_t idx2 = map.getDimPosition(i);
            if (idx2 == 0) {
              perm.insert(perm.begin(), i);
              auto targetExpr = rewriter.getAffineDimExpr(idx2);
              transposeResults.insert(transposeResults.begin(), targetExpr);
            } else {
              perm.push_back(i);
              auto targetExpr = rewriter.getAffineDimExpr(idx2);
              transposeResults.push_back(targetExpr);
            }
          }
          map = AffineMap::get(map.getNumDims(), 0, transposeResults,
                               contractOp.getContext());
          rhs = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), rhs,
                                                     perm);
        }
        if (it.index() == 2) // acc opernand has a transposition in the leading
                             // Dim, we dont support this
          return failure();
      }
      if (map.getDimPosition(0) == 0 && it.index() < 2)
        validExtract[it.index()] = true;

      for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
        int64_t idx = map.getDimPosition(i);
        if (idx == 0) // This is the dimension we are removing
          continue;
        auto targetExpr = rewriter.getAffineDimExpr(idx - 1);
        results.push_back(targetExpr);
      }
      newIndexingMaps.push_back(AffineMap::get(map.getNumDims() - 1, 0, results,
                                               contractOp.getContext()));
    }
    auto newLhs = validExtract[0]
                      ? rewriter.create<vector::ExtractOp>(
                            contractOp.getLoc(), lhs, splatZero(dropDim))
                      : lhs;
    auto newRhs = validExtract[1]
                      ? rewriter.create<vector::ExtractOp>(
                            contractOp.getLoc(), rhs, splatZero(dropDim))
                      : rhs;
    auto newAcc = rewriter.create<vector::ExtractOp>(
        contractOp.getLoc(), contractOp.acc(), splatZero(dropDim));
    auto newContractOp = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), newLhs, newRhs, newAcc,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        rewriter.getArrayAttr(newIteratorTypes));
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        contractOp, contractOp->getResultTypes()[0], newContractOp);
    return success();
  }
};

class CastAwayElementwiseLeadingOneDim : public RewritePattern {
public:
  CastAwayElementwiseLeadingOneDim(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>();
    if (!vecType)
      return failure();
    VectorType newVecType = trimLeadingOneDims(vecType);
    if (newVecType == vecType)
      return failure();
    int64_t dropDim = vecType.getRank() - newVecType.getRank();
    SmallVector<Value, 4> newOperands;
    for (Value operand : op->getOperands()) {
      if (auto opVecType = operand.getType().dyn_cast<VectorType>()) {
        newOperands.push_back(rewriter.create<vector::ExtractOp>(
            op->getLoc(), operand, splatZero(dropDim)));
      } else {
        newOperands.push_back(operand);
      }
    }
    OperationState state(op->getLoc(), op->getName());
    state.addAttributes(op->getAttrs());
    state.addOperands(newOperands);
    state.addTypes(newVecType);
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecType,
                                                     newOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<CastAwayExtractStridedSliceLeadingOneDim,
           CastAwayInsertStridedSliceLeadingOneDim,
           CastAwayTransferReadLeadingOneDim,
           CastAwayTransferWriteLeadingOneDim, CastAwayElementwiseLeadingOneDim,
           CastAwayContractionLeadingOneDim>(patterns.getContext());
  populateShapeCastFoldingPatterns(patterns);
}
