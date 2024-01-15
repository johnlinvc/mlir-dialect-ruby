add_mlir_dialect_library(MLIRRubyIseq
        RubyIseqTypes.cpp
        RubyIseqDialect.cpp
        RubyIseqOps.cpp
        RubyIseqPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RubyIseq

        DEPENDS
        MLIRRubyIseqOpsIncGen
        MLIRRubyIseqPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
