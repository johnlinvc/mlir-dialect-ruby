# frozen_string_literal: true

require "test_helper"

module Mlir
  module Dialect
    class TestRuby < Minitest::Test
      def test_that_it_has_a_version_number
        refute_nil ::Mlir::Dialect::Ruby::VERSION
      end
    end
  end
end
