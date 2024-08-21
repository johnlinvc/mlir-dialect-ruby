//===- RubyPasses.cpp - Ruby passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "Ruby/RubyPasses.h"

namespace mlir::ruby {
#define GEN_PASS_DEF_RUBYSWITCHBARFOO
#define GEN_PASS_DEF_RUBYCALLTOARITH
#define GEN_PASS_DEF_RUBYTYPEINFER
#include "Ruby/RubyPasses.h.inc"
#include "Ruby/RubyPatterns.h.inc"

namespace {
class RubySwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class RubySwitchBarFoo
    : public impl::RubySwitchBarFooBase<RubySwitchBarFoo> {
public:
  using impl::RubySwitchBarFooBase<
      RubySwitchBarFoo>::RubySwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RubySwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

namespace {
class RubyRewriteLocalVarWriteRetType : public OpRewritePattern<LocalVariableWriteOp> {
public:
  using OpRewritePattern<LocalVariableWriteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LocalVariableWriteOp op,
                                PatternRewriter &rewriter) const final {
    auto inputOp = op.getInput().getDefiningOp<CastOp>();
    if (!inputOp) {
      return failure();
    }
    auto newOp = rewriter.create<LocalVariableWriteOp>(
        op.getLoc(), inputOp.getInput().getType(), op.getVarName(), inputOp.getInput());
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, newOp.getOperation());
    
    return success();
  }
};
}

class RubyTypeInfer :
 public impl::RubyTypeInferBase<RubyTypeInfer> {
  public:
  using impl::RubyTypeInferBase<RubyTypeInfer>::RubyTypeInferBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RubyRewriteLocalVarWriteRetType>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

namespace {
class RubyRewriteCallWithAdd : public OpRewritePattern<CallOp> {
public:
  using OpRewritePattern<CallOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const final {
    auto methodName = op.getMethodName();
    if (methodName != "+") {
      return failure();
    }
    if (op.getArgs().size() != 1) {
      return failure();
    }
    auto lhs = op.getCallee();
    if (lhs == ::mlir::Value()) {
      return failure();
    }
    auto rhs = op.getArgs()[0];
    if (!lhs.getType().isa<ruby::IntegerType>() ||
        !rhs.getType().isa<ruby::IntegerType>()) {
      return failure();
    }
    auto resultType = lhs.getType();
    auto newAdd = rewriter.create<AddOp>(op.getLoc(), resultType, lhs, rhs);
    rewriter.replaceOpWithNewOp<CastOp>(
        op, op.getRes().getType(), newAdd);
    
    return success();
  }
};
}


class RubyCallToArith :
 public impl::RubyCallToArithBase<RubyCallToArith> {
  public:
  using impl::RubyCallToArithBase<RubyCallToArith>::RubyCallToArithBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RubyRewriteCallWithAdd>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::ruby
