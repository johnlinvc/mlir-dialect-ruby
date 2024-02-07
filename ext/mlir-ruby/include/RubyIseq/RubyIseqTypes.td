//===- RubyIseqTypes.td - RubyIseq dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBYISEQ_TYPES
#define RUBYISEQ_TYPES

include "mlir/IR/AttrTypeBase.td"
include "RubyIseq/RubyIseqDialect.td"

//===----------------------------------------------------------------------===//
// RubyIseq type definitions
//===----------------------------------------------------------------------===//

class RubyIseq_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RubyIseq_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RubyIseq_CustomType : RubyIseq_Type<"Custom", "custom"> {
    let summary = "RubyIseq custom type";
    let description = "Custom type in rubyiseq dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

def RubyIseq_IntegerType : RubyIseq_Type<"Integer", "int"> {
    let summary = "RubyIseq integer type";
    let description = "Integer type in rubyiseq dialect";
    let parameters = (ins DefaultValuedParameter<"StringAttr", "$_builder.getStringAttr(\"0\")">:$value);
    let assemblyFormat = "`<` $value `>`";
}

def RubyIseq_AnyType : AnyTypeOf<[RubyIseq_IntegerType]>;

#endif // RUBYISEQ_TYPES
