//===- RubyOps.cpp - Ruby dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ruby/RubyOps.h"
#include "Ruby/RubyDialect.h"
#include "mlir/Dialect/CommonFolders.h"

#define GET_OP_CLASSES
#include "Ruby/RubyOps.cpp.inc"

namespace mlir {
    namespace ruby
    {
        OpFoldResult ConstantIntOp::fold(ConstantIntOp::FoldAdaptor adaptor) {
            return Value();
        }
        OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){
            auto operands = adaptor.getOperands();
            return operands[0];
        }

    

        // }
    } // namespace ruby
} // namespace mlir
