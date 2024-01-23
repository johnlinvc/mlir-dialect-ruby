# frozen_string_literal: true

require "test_helper"

describe MLIR::Dialect::Ruby do
  it "gets or loads dialect" do
    context = MLIR::CAPI.mlirContextCreate
    MLIR::CAPI.register_all_upstream_dialects(context)
    MLIR::CAPI.mlirContextGetOrLoadDialect(context, MLIR::CAPI.mlirStringRefCreateFromCString("arith"))
    MLIR::CAPI.mlirContextDestroy(context)
  end
  it "creates a dialect handle" do
    MLIR::Dialect::Ruby::CAPI.mlirGetDialectHandle__ruby__
  end

  it "pass the default test in ruby-capi-test.c" do
    # line 26
    context = MLIR::CAPI.mlirContextCreate

    # line 29
    MLIR::CAPI.register_all_upstream_dialects(context)
    MLIR::CAPI.mlirContextGetOrLoadDialect(context, MLIR::CAPI.mlirStringRefCreateFromCString("arith"))
    # line 30
    MLIR::CAPI.mlirDialectHandleRegisterDialect(MLIR::Dialect::Ruby::CAPI.mlirGetDialectHandle__ruby__, context)

    # line 32
    str = "%0 = arith.constant 2 : i32\n%1 = ruby.foo %0 : i32\n"
    mlir_str = MLIR::CAPI.mlirStringRefCreateFromCString(str)
    module1 = MLIR::CAPI.mlirModuleCreateParse(context, mlir_str)

    op = MLIR::CAPI.mlirModuleGetOperation(module1)
    MLIR::CAPI.mlirOperationDump(op)

    MLIR::CAPI.mlirContextDestroy(context)
  end
end
