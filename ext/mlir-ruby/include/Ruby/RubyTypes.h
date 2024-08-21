//===- RubyTypes.h - Ruby dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBY_RUBYTYPES_H
#define RUBY_RUBYTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_TYPEDEF_CLASSES
#include "Ruby/RubyOpsTypes.h.inc"

#endif // RUBY_RUBYTYPES_H
