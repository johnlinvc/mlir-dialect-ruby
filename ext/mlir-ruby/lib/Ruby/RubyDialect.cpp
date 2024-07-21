//===- RubyDialect.cpp - Ruby dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ruby/RubyDialect.h"
#include "Ruby/RubyOps.h"
#include "Ruby/RubyTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ruby;

#include "Ruby/RubyOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Ruby dialect.
//===----------------------------------------------------------------------===//

void RubyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ruby/RubyOps.cpp.inc"
      >();
  registerTypes();
}

/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation *RubyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  Operation * result =
      llvm::TypeSwitch<Type, Operation *>(type)
          .Case<IntegerType>([&](auto type)
                          { 
                            llvm::dbgs() << "materializeConstant: IntegerType\n";
                            auto strAttr = value.dyn_cast<StringAttr>();
                            if (!strAttr) {
                              llvm::dbgs() << "strAttr is null\n";
                            }
                            return strAttr ? builder.create<ConstantIntOp>(loc, type, strAttr) : nullptr; 
                          })
          .Default([&](auto type)
                          { return nullptr; });
  return result;
}