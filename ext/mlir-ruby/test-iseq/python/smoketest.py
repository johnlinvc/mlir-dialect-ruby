# RUN: %python %s | FileCheck %s

from mlir_rubyiseq.ir import *
from mlir_rubyiseq.dialects import builtin as builtin_d, rubyiseq as rubyiseq_d

with Context():
    rubyiseq_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = rubyiseq.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: rubyiseq.foo %[[C]] : i32
    print(str(module))
