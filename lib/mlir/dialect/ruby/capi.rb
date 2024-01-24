# frozen_string_literal: true

require "ffi"
module MLIR
  module Dialect
    module Ruby
      # CAPI wrapper
      module CAPI
        extend FFI::Library
        ffi_lib "RubyAllCAPILib"
        attach_function :mlirGetDialectHandle__ruby__, [], MLIR::CAPI::MlirDialectHandle.by_value
      end
    end
  end
end
