# frozen_string_literal: true

require "mlir"
require_relative "ruby/version"
require_relative "ruby/capi"
require_relative "ruby/prism_loader"
require_relative "ruby/prism_optimizer"

module MLIR
  module Dialect
    module Ruby
      class Error < StandardError; end
      # Your code goes here...
    end
  end
end
