add_mlir_public_c_api_library(RubyIseqCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRRubyIseq
)
