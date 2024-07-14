add_mlir_dialect_library(MLIRRuby
        RubyTypes.cpp
        RubyDialect.cpp
        RubyOps.cpp
        RubyPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Ruby

        DEPENDS
        MLIRRubyOpsIncGen
        MLIRRubyPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
add_subdirectory(Target)
