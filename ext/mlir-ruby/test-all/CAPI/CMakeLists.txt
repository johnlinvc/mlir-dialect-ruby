# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own ruby dialect.
add_mlir_aggregate(RubyAllCAPILib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  RubyCAPI
  RubyIseqCAPI
)

add_llvm_executable(ruby-all-capi-test
  ruby-all-capi-test.c
)
llvm_update_compile_flags(ruby-all-capi-test)
target_link_libraries(ruby-all-capi-test
  PRIVATE RubyAllCAPILib)
