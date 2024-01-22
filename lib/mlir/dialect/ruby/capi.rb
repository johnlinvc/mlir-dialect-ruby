# frozen_string_literal: true
require 'ffi'
module MLIR
  module Dialect
    module Ruby
        module CAPI
            extend FFI::Library
            ffi_lib "RubyCAPILib"
            attach_function :mlirGetDialectHandle__ruby__,[], MLIR::CAPI::MlirDialectHandle.by_value 
        end
    end
  end
end
