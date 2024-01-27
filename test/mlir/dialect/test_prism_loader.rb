# frozen_string_literal: true

require "test_helper"

describe MLIR::Dialect::Ruby::PrismLoader do
  it "loads '1+1' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+1")
    loader.to_module
  end
  it "loads '1+(1+1)' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+(1+1)")
    loader.to_module
  end
  it "loads '1+2+3' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+1+1")
    loader.to_module
  end
end
