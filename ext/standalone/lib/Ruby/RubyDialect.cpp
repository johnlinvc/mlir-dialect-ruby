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
