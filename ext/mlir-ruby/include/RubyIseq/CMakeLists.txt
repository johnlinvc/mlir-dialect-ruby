add_mlir_dialect(RubyIseqOps rubyiseq)
add_mlir_doc(RubyIseqDialect RubyIseqDialect RubyIseq/ -gen-dialect-doc)
add_mlir_doc(RubyIseqOps RubyIseqOps RubyIseq/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RubyIseqPasses.td)
mlir_tablegen(RubyIseqPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRRubyIseqPassesIncGen)
