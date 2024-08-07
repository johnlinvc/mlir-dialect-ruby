//===- RubyIseqPsss.td - RubyIseq dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBYISEQ_PASS
#define RUBYISEQ_PASS

include "mlir/Pass/PassBase.td"

def RubyIseqSwitchBarFoo: Pass<"rubyiseq-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def RubyIseqLowerFromRuby: Pass<"rubyiseq-lower-from-ruby", "::mlir::ModuleOp"> {
  let summary = "Lowers a RubyIseq dialect module to the LLVM dialect.";
  let description = [{
    Lowers a RubyIseq dialect module to the LLVM dialect.
  }];
}

#endif // RUBYISEQ_PASS
