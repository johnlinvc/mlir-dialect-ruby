//===- RubyIseqOps.h - RubyIseq dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBYISEQ_RUBYISEQOPS_H
#define RUBYISEQ_RUBYISEQOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "RubyIseq/RubyIseqTypes.h"

#define GET_OP_CLASSES
#include "RubyIseq/RubyIseqOps.h.inc"

#endif // RUBYISEQ_RUBYISEQOPS_H
