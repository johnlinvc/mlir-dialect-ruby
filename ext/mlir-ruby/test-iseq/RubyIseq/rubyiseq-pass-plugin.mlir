// RUN: mlir-opt %s --load-pass-plugin=%rubyiseq_libs/RubyIseqPlugin%shlibext --pass-pipeline="builtin.module(rubyiseq-switch-bar-foo)" | FileCheck %s

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
