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
end
