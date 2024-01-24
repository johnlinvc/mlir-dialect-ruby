llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RUBY_ALL_TEST_DEPENDS
        FileCheck count not
        ruby-all-capi-test
        )

add_lit_testsuite(check-ruby-all "Running the ruby regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RUBY_ALL_TEST_DEPENDS}
        )

set_target_properties(check-ruby-all PROPERTIES FOLDER "Tests")

add_lit_testsuites(RUBY_ALL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RUBY_ALL_TEST_DEPENDS})

add_subdirectory(CAPI)
