//===- RubyIseqPasses.h - RubyIseq passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RUBYISEQ_RUBYISEQPASSES_H
#define RUBYISEQ_RUBYISEQPASSES_H

#include "RubyIseq/RubyIseqDialect.h"
#include "RubyIseq/RubyIseqOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace rubyiseq {
#define GEN_PASS_DECL
#include "RubyIseq/RubyIseqPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RubyIseq/RubyIseqPasses.h.inc"
} // namespace rubyiseq
} // namespace mlir

#endif
