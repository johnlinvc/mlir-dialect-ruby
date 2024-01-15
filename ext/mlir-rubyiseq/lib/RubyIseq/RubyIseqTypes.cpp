//===- RubyIseqTypes.cpp - RubyIseq dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RubyIseq/RubyIseqTypes.h"

#include "RubyIseq/RubyIseqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::rubyiseq;

#define GET_TYPEDEF_CLASSES
#include "RubyIseq/RubyIseqOpsTypes.cpp.inc"

void RubyIseqDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RubyIseq/RubyIseqOpsTypes.cpp.inc"
      >();
}
