# frozen_string_literal: true

require "test_helper"

describe MLIR::Dialect::Ruby::PrismLoader do
  include MLIRHelper
  it "loads '1+1' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+1")
    mod = loader.to_module
    parse_with_opt(mod)
  end
  it "loads '1+(1+1)' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+(1+1)")
    mod = loader.to_module
    parse_with_opt(mod)
  end
  it "loads '1+2+3' " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("1+2+3")
    mod = loader.to_module
    parse_with_opt(mod)
  end
  it "loads set local variable " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("a=42")
    mod = loader.to_module
    parse_with_opt(mod)
  end

  it "loads get local variable " do
    loader = MLIR::Dialect::Ruby::PrismLoader.new("a=42\na")
    mod = loader.to_module
    parse_with_opt(mod)
  end
end
