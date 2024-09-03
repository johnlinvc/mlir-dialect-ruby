// RUN: mlir-opt %s --load-pass-plugin=%ruby_libs/RubyPlugin%shlibext --pass-pipeline="builtin.module()" | FileCheck %s

module {
  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
