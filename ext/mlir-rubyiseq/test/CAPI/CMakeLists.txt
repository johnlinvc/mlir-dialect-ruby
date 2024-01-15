# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own rubyiseq dialect.
add_mlir_aggregate(RubyIseqCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  RubyIseqCAPI
)

add_llvm_executable(rubyiseq-capi-test
  rubyiseq-capi-test.c
)
llvm_update_compile_flags(rubyiseq-capi-test)
target_link_libraries(rubyiseq-capi-test
  PRIVATE RubyIseqCAPITestLib)
