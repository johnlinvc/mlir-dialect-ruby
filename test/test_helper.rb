# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
ENV["MLIR_LIB_NAME"] = "RubyAllCAPILib"
require "bundler/setup"
require "mlir"
require "mlir/dialect/ruby"

require "minitest/autorun"
