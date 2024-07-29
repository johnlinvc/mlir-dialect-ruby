# frozen_string_literal: true

require "prism"
require "mlir"
require "erb"

module MLIR
  module Dialect
    module Ruby
      # An optimizer using mlir
      class Optimizer
        def initialize(loader_klass = PrismLoader)
          @loader_klass = loader_klass
        end

        def optimize(program)
          loader = @loader_klass.new(program)
          mod = loader.to_module
          puts mod
          optimzed_mod = optimize_with_opt(mod)
          puts optimzed_mod
          translate_back(optimzed_mod)
        end

        def translate_back(mod)
          Tempfile.create("optimized_rb") do |f|
            f.write(mod)
            f.close
            translate_cmd = File.expand_path("../../../../ext/mlir-ruby/build/bin/ruby-translate", __dir__)
            out = `#{translate_cmd} --mlir-to-ruby #{f.path}`
            raise "opt failed: #{out}" unless $CHILD_STATUS.success?

            out
          end
        end

        def optimize_with_opt(mod)
          Tempfile.create("test") do |f|
            f.write(mod)
            f.close
            opt_cmd = File.expand_path("../../../../ext/mlir-ruby/build/bin/ruby-opt", __dir__)
            out = `#{opt_cmd} --ruby-call-to-arith --ruby-type-infer --canonicalize #{f.path}`
            raise "opt failed: #{out}" unless $CHILD_STATUS.success?

            puts "mlir after opt"
            system("cat #{f.path}")

            out
          end
        end
      end
    end
  end
end
