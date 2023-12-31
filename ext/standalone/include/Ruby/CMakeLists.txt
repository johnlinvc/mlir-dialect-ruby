add_mlir_dialect(RubyOps ruby)
add_mlir_doc(RubyDialect RubyDialect Ruby/ -gen-dialect-doc)
add_mlir_doc(RubyOps RubyOps Ruby/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RubyPasses.td)
mlir_tablegen(RubyPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRRubyPassesIncGen)
