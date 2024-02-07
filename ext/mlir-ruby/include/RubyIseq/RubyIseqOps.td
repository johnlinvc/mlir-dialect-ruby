//===- RubyIseqOps.td - RubyIseq dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUBYISEQ_OPS
#define RUBYISEQ_OPS

include "RubyIseq/RubyIseqTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def RubyIseq_opt_plusOP : RubyIseq_Op<"opt_plus", []> {
    let summary = "Add two integers.";
    let description = [{
        The `rubyiseq.opt_plus` operation adds two integers. It takes two
        integers as arguments and returns an integer.

        Example:

        ```mlir
        rubyiseq.opt_plus
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let assemblyFormat = [{ attr-dict }];
}

def RubyIseq_leaveOP : RubyIseq_Op<"leave", []> {
    let summary = "Leave the current control frame.";
    let description = [{
        The `rubyiseq.leave` operation leaves the current control frame. It
        takes no arguments and returns no results.

        Example:

        ```mlir
        // Leave the current control frame
        %0 = rubyiseq.leave
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let assemblyFormat = [{ attr-dict }];
}

def RubyIseq_putobjectOP : RubyIseq_Op<"putobject", []> {
    let summary = "Put an object on the stack.";
    let description = [{
        The `rubyiseq.putobject` operation puts an object on the stack. It
        takes an object as an argument and returns the object.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Put the object %0 on the stack
        %1 = rubyiseq.putobject %0 : i32
        ```
    }];

    let arguments = (ins RubyIseq_AnyType:$input);
    let results = (outs);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict 
    }];
}

def RubyIseq_FooOp : RubyIseq_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `rubyiseq.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = rubyiseq.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // RUBYISEQ_OPS
