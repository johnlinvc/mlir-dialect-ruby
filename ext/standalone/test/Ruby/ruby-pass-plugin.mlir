// RUN: mlir-opt %s --load-pass-plugin=%ruby_libs/RubyPlugin%shlibext --pass-pipeline="builtin.module(ruby-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
