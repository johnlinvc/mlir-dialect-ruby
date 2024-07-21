//===- RubyOps.td - Ruby dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBY_OPS
#define RUBY_OPS

include "Ruby/RubyTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Ruby_LocalVariableReadOp : Ruby_Op<"local_variable_read", []> {
    let arguments = (ins StrAttr:$var_name);
    let results = (outs Ruby_AnyType:$res);

    let assemblyFormat = [{
        $var_name attr-dict `:` type($res)
    }];
}

def Ruby_LocalVariableWriteOp : Ruby_Op<"local_variable_write", [SameOperandsAndResultType]> {
    let arguments = (ins StrAttr:$var_name, Ruby_AnyType:$input);
    let results = (outs Ruby_AnyType:$res);

    let assemblyFormat = [{
        $var_name `=` $input attr-dict `:` type($res)
    }];
}

def Ruby_ConstantStrOp : Ruby_Op<"constant_str", []> {
    let arguments = (ins StrAttr:$input);
    let results = (outs Ruby_StringType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($res)
    }];
}

def Ruby_CallOp : Ruby_Op<"call", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Ruby_AnyType>:$callee, StrAttr:$method_name, Variadic<Ruby_AnyType>:$args);
    let results = (outs Ruby_AnyType:$res);

    let assemblyFormat = [{
        ($callee^ `:` type($callee))? `->` $method_name `(` ($args^)? `)` attr-dict `:` functional-type($args, $res)
    }];
}

class Ruby_BinaryOp<string mnemonic> : Ruby_Op<mnemonic, [Pure]> {
    let arguments = (ins Ruby_AnyType:$lhs, Ruby_AnyType:$rhs);
    let results = (outs Ruby_AnyType:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res) 
    }];
}

def Ruby_AddOp : Ruby_BinaryOp<"add"> {
    let summary = "Add two objects";
    let description = [{
        This operation adds two objects.
    }];
    let hasFolder = 1;
}

def Ruby_ConstantIntOp : Ruby_Op<"constant_int", []> {
    let arguments = (ins StrAttr:$input);
    let results = (outs Ruby_IntegerType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($res)
    }];
    let hasFolder = 1;
}

def Ruby_ReturnOp : Ruby_Op<"return", [Terminator]> {
    let arguments = (ins Optional<Ruby_AnyType>:$input);
    let assemblyFormat = [{
        attr-dict ($input^ `:` type($input))?
    }];
}

def Ruby_DefOp : Ruby_Op<"def", []> {
    let arguments = (ins StrAttr:$method_name, Optional<Ruby_AnyType>:$receiver ,
     OptionalAttr<StrArrayAttr>:$required_args, 
     OptionalAttr<TypeArrayAttr>:$required_arg_types,
     TypeAttr:$ret_type
     );
    let regions = (region AnyRegion:$body);
    let results = (outs Ruby_SymbolType:$res);

    let assemblyFormat = [{
        $method_name (`+``(`$receiver^ `:` type($receiver)`)`)? 
        `(`
            (`required_args` `:` $required_args^)?
        `)` attr-dict `:` `(` 
            (`required_args``:`$required_arg_types^)?
        `)` `->` $ret_type 
        $body `:` type($res)
    }];
}

def Ruby_FooOp : Ruby_Op<"foo", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `ruby.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = ruby.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // RUBY_OPS
