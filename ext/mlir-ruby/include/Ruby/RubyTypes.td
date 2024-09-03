//===- RubyTypes.td - Ruby dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBY_TYPES
#define RUBY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Ruby/RubyDialect.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

//===----------------------------------------------------------------------===//
// Ruby type definitions
//===----------------------------------------------------------------------===//

class Ruby_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ruby_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ruby_StringType : Ruby_Type<"String", "string"> {
    let summary = "Ruby string type";
    let description = "String type in ruby dialect";
}

def Ruby_IntegerType : Ruby_Type<"Integer", "int"> {
    let summary = "Ruby integer type";
    let description = "Integer type in ruby dialect";
}

def Ruby_SymbolType : Ruby_Type<"Symbol", "sym"> {
    let summary = "Ruby symbol type";
    let description = "Symbol type in ruby dialect";
}

def Ruby_OpaqueObjectType : Ruby_Type<"OpaqueObject", "opaque_object"> {
    let summary = "Ruby opaque object type";
    let description = "Opaque object type in ruby dialect";
}

def Ruby_AnyType : AnyTypeOf<[Ruby_IntegerType, Ruby_StringType, Ruby_OpaqueObjectType, Ruby_SymbolType]>;

#endif // RUBY_TYPES
