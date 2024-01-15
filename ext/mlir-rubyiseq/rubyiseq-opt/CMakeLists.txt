get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRRubyIseq
        )
add_llvm_executable(rubyiseq-opt rubyiseq-opt.cpp)

llvm_update_compile_flags(rubyiseq-opt)
target_link_libraries(rubyiseq-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(rubyiseq-opt)
