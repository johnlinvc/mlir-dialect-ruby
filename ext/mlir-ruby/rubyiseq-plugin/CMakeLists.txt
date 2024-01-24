add_llvm_library(RubyIseqPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        rubyiseq-plugin.cpp

        DEPENDS
        MLIRRubyIseq
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRRubyIseq
        )
target_include_directories(
        RubyIseqPlugin
        PRIVATE
        "${RUBYISEQ_BINARY_DIR}/include"
)
