
# frozen_string_literal: true

require "test_helper"

describe MLIR::Dialect::Ruby::Optimizer do
  include MLIRHelper
  it "loads '1+1' " do
    optimizer = MLIR::Dialect::Ruby::Optimizer.new
    result = optimizer.optimize("1+1")
    result.must_equal("2")
  end
end