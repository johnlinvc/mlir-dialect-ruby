//===- RubyDialect.td - Ruby dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBY_DIALECT
#define RUBY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Ruby dialect definition.
//===----------------------------------------------------------------------===//

def Ruby_Dialect : Dialect {
    let name = "ruby";
    let summary = "A ruby out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::ruby";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base ruby operation definition.
//===----------------------------------------------------------------------===//

class Ruby_Op<string mnemonic, list<Trait> traits = []> :
        Op<Ruby_Dialect, mnemonic, traits>;

#endif // RUBY_DIALECT
