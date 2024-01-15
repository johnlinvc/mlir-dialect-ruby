# RUN: %python %s | FileCheck %s

from mlir_ruby.ir import *
from mlir_ruby.dialects import builtin as builtin_d, ruby as ruby_d

with Context():
    ruby_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = ruby.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: ruby.foo %[[C]] : i32
    print(str(module))
