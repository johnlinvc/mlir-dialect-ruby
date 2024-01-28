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
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+2+3")
    loader.to_module
  end
  it "loads set local variable " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("a=42")
    loader.to_module
  end

  it "loads get local variable " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("a=42\na")
    loader.to_module
  end
end
