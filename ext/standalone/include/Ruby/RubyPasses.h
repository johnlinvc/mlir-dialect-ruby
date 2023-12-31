//===- RubyPasses.h - Ruby passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RUBY_RUBYPASSES_H
#define RUBY_RUBYPASSES_H

#include "Ruby/RubyDialect.h"
#include "Ruby/RubyOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace ruby {
#define GEN_PASS_DECL
#include "Ruby/RubyPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Ruby/RubyPasses.h.inc"
} // namespace ruby
} // namespace mlir

#endif
