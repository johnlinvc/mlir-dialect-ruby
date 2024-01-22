# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own ruby dialect.
add_mlir_aggregate(RubyCAPILib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  RubyCAPI
)

add_llvm_executable(ruby-capi-test
  ruby-capi-test.c
)
llvm_update_compile_flags(ruby-capi-test)
target_link_libraries(ruby-capi-test
  PRIVATE RubyCAPILib)
