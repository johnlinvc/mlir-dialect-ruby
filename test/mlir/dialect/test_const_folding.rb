# frozen_string_literal: true

require "test_helper"

describe MLIR::Dialect::Ruby::Optimizer do
  include MLIRHelper
  it "loads '1+1' " do
    optimizer = MLIR::Dialect::Ruby::Optimizer.new
    result = optimizer.optimize("coscup = 40 + 2")
    _(result).must_equal("coscup = 42\n")
  end
end
