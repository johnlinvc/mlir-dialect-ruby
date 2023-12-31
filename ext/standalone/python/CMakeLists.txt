include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_ruby`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_ruby.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(RubyPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT RubyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ruby"
  TD_FILE dialects/RubyOps.td
  SOURCES
    dialects/ruby.py
  DIALECT_NAME ruby)

declare_mlir_python_extension(RubyPythonSources.Extension
  MODULE_NAME _rubyDialects
  ADD_TO_PARENT RubyPythonSources
  SOURCES
    RubyExtension.cpp
  EMBED_CAPI_LINK_LIBS
    RubyCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(RubyPythonCAPI
  INSTALL_COMPONENT RubyPythonModules
  INSTALL_DESTINATION python_packages/ruby/mlir_ruby/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/ruby/mlir_ruby/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    RubyPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(RubyPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/ruby/mlir_ruby"
  INSTALL_PREFIX "python_packages/ruby/mlir_ruby"
  DECLARED_SOURCES
    RubyPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    RubyPythonCAPI
  )
