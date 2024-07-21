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

#include <iostream>

#define GET_OP_CLASSES
#include "Ruby/RubyOps.cpp.inc"

namespace mlir {
    namespace ruby
    {
        OpFoldResult ConstantIntOp::fold(ConstantIntOp::FoldAdaptor adaptor) {
            return StringAttr::get(getContext(), adaptor.getInput());
        }

    //     OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){
    //         return constFoldBinaryOp<StringAttr>(
    //   adaptor.getOperands(), [&](auto a, auto b) { return a;  });
        // }
        OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){
            auto operands = adaptor.getOperands();
            std::cerr << "AddOp::fold" << std::endl << operands.size() << std::endl;
            if (operands[0] == mlir::Attribute())
                std::cerr << "operands[0] is null" << std::endl;
            if (isa<mlir::StringAttr>(operands[0]))
                std::cerr << "operands[0] is StringAttr" << std::endl;
            if (isa<mlir::Attribute>(operands[0]))
                std::cerr << "operands[0] is mlir::Attribute" << std::endl;
            // llvm::outs() << operands[0] << "\n";
            if (!operands[0] || !operands[1])
                return nullptr;
            return operands[0];
            auto strLhs = operands[0].dyn_cast<StringAttr>();
            if (!strLhs)
              std::cerr << "strLhs is null" << std::endl;
            return nullptr;
            auto lhs = strLhs.getValue();
            auto lhsInt = ::std::stoi(lhs.str());
            auto strRhs = operands[1].dyn_cast<StringAttr>();
            auto rhs = strRhs.getValue();
            auto rhsInt = ::std::stoi(rhs.str());
            auto sum = lhsInt + rhsInt;
            auto sumStr = ::std::to_string(sum);
            auto str = StringAttr::get(getContext(), sumStr);
            return str;
        }

    

        // }
    } // namespace ruby
} // namespace mlir
