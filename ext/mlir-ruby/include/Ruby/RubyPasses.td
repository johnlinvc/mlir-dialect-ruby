//===- RubyPsss.td - Ruby dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBY_PASS
#define RUBY_PASS

include "mlir/Pass/PassBase.td"

def RubySwitchBarFoo: Pass<"ruby-switch-bar-foo", "::mlir::ModuleOp"> {
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

def RubyCallToArith: Pass<"ruby-call-to-arith", "::mlir::ModuleOp"> {
  let summary = "Replace call operations with arith operations.";
  let description = [{
    Change call operations to arith operations.
    ```
    ruby.call $1:!ruby.int, "+", ($2) : (!ruby.int) -> !ruby.int
    // Gets transformed to:
    ruby.add $1, $2 : (!ruby.int, !ruby.int) -> !ruby.int
    ```
  }];
}

#endif // RUBY_PASS
