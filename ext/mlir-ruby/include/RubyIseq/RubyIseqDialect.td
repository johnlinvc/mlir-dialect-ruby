//===- RubyIseqDialect.td - RubyIseq dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBYISEQ_DIALECT
#define RUBYISEQ_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RubyIseq dialect definition.
//===----------------------------------------------------------------------===//

def RubyIseq_Dialect : Dialect {
    let name = "rubyiseq";
    let summary = "A rubyiseq out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::rubyiseq";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base rubyiseq operation definition.
//===----------------------------------------------------------------------===//

class RubyIseq_Op<string mnemonic, list<Trait> traits = []> :
        Op<RubyIseq_Dialect, mnemonic, traits>;

#endif // RUBYISEQ_DIALECT
