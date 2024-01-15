// RUN: mlir-opt %s --load-dialect-plugin=%ruby_libs/RubyPlugin%shlibext --pass-pipeline="builtin.module(ruby-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @ruby_types(%arg0: !ruby.custom<"10">)
  func.func @ruby_types(%arg0: !ruby.custom<"10">) {
    return
  }
}
