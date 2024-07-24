# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
ENV["MLIR_LIB_NAME"] = "RubyAllCAPILib"
require "bundler/setup"
require "debug"
require "mlir"
require "mlir/dialect/ruby"

require "minitest/autorun"
require "english"

module MLIRHelper
  def parse_with_opt(mod)
    Tempfile.create("test") do |f|
      f.write(mod)
      f.close
      opt_cmd = File.expand_path("../ext/mlir-ruby/build/bin/ruby-opt", __dir__)
      out = `#{opt_cmd} #{f.path}`
      raise "opt failed: #{out}" unless $CHILD_STATUS.success?

      out
    end
  end
end
