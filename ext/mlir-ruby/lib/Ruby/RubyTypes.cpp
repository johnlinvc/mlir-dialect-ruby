//===- RubyTypes.cpp - Ruby dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ruby/RubyTypes.h"

#include "Ruby/RubyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::ruby;

#define GET_TYPEDEF_CLASSES
#include "Ruby/RubyOpsTypes.cpp.inc"

void RubyDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Ruby/RubyOpsTypes.cpp.inc"
      >();
}
