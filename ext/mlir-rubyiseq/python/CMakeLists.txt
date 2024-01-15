include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_rubyiseq`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_rubyiseq.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(RubyIseqPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT RubyIseqPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_rubyiseq"
  TD_FILE dialects/RubyIseqOps.td
  SOURCES
    dialects/rubyiseq.py
  DIALECT_NAME rubyiseq)

declare_mlir_python_extension(RubyIseqPythonSources.Extension
  MODULE_NAME _rubyiseqDialects
  ADD_TO_PARENT RubyIseqPythonSources
  SOURCES
    RubyIseqExtension.cpp
  EMBED_CAPI_LINK_LIBS
    RubyIseqCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(RubyIseqPythonCAPI
  INSTALL_COMPONENT RubyIseqPythonModules
  INSTALL_DESTINATION python_packages/rubyiseq/mlir_rubyiseq/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/rubyiseq/mlir_rubyiseq/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    RubyIseqPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(RubyIseqPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/rubyiseq/mlir_rubyiseq"
  INSTALL_PREFIX "python_packages/rubyiseq/mlir_rubyiseq"
  DECLARED_SOURCES
    RubyIseqPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    RubyIseqPythonCAPI
  )
