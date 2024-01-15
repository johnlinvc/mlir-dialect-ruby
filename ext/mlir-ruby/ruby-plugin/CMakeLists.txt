add_llvm_library(RubyPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        ruby-plugin.cpp

        DEPENDS
        MLIRRuby
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRRuby
        )
target_include_directories(
        RubyPlugin
        PRIVATE
        "${RUBY_BINARY_DIR}/include"
)
