llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RUBY_TEST_DEPENDS
        FileCheck count not
        ruby-capi-test
        ruby-opt
        ruby-translate
        RubyPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND RUBY_TEST_DEPENDS RubyPythonModules)
endif()

add_lit_testsuite(check-ruby "Running the ruby regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RUBY_TEST_DEPENDS}
        )
set_target_properties(check-ruby PROPERTIES FOLDER "Tests")

add_lit_testsuites(RUBY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RUBY_TEST_DEPENDS})

add_subdirectory(CAPI)
