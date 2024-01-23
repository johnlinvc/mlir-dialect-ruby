//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ruby-c/Dialects.h"

#include "Ruby/RubyDialect.h"
#include "RubyIseq/RubyIseqDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ruby, ruby,
                                      mlir::ruby::RubyDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RubyIseq, rubyiseq,
                                      mlir::rubyiseq::RubyIseqDialect)