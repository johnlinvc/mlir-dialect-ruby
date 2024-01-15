get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRRuby
        )
add_llvm_executable(ruby-opt ruby-opt.cpp)

llvm_update_compile_flags(ruby-opt)
target_link_libraries(ruby-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ruby-opt)
