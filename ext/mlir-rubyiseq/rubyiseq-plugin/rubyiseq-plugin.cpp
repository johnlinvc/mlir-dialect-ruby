//===- rubyiseq-plugin.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "RubyIseq/RubyIseqDialect.h"
#include "RubyIseq/RubyIseqPasses.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "RubyIseq", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::rubyiseq::RubyIseqDialect>();
            mlir::rubyiseq::registerPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "RubyIseqPasses", LLVM_VERSION_STRING,
          []() { mlir::rubyiseq::registerPasses(); }};
}
