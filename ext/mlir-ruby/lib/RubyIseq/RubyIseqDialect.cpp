//===- RubyIseqDialect.cpp - RubyIseq dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RubyIseq/RubyIseqDialect.h"
#include "RubyIseq/RubyIseqOps.h"
#include "RubyIseq/RubyIseqTypes.h"

using namespace mlir;
using namespace mlir::rubyiseq;

#include "RubyIseq/RubyIseqOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RubyIseq dialect.
//===----------------------------------------------------------------------===//

void RubyIseqDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RubyIseq/RubyIseqOps.cpp.inc"
      >();
  registerTypes();
}
