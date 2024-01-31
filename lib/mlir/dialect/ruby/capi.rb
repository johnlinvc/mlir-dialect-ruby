# frozen_string_literal: true

require "ffi"
module MLIR
  module Dialect
    module Ruby
      FFI_LIB_NAME = ENV["MLIR_RUBY_LIB_NAME"] || "RubyAllCAPILib"
      # CAPI wrapper
      module CAPI
        extend FFI::Library
        ffi_lib MLIR::Dialect::Ruby::FFI_LIB_NAME
        attach_function :mlirGetDialectHandle__ruby__, [], MLIR::CAPI::MlirDialectHandle.by_value
      end
    end
  end
end
