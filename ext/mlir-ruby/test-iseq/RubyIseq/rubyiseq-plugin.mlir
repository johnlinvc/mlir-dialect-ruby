// RUN: mlir-opt %s --load-dialect-plugin=%rubyiseq_libs/RubyIseqPlugin%shlibext --pass-pipeline="builtin.module(rubyiseq-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @rubyiseq_types(%arg0: !rubyiseq.custom<"10">)
  func.func @rubyiseq_types(%arg0: !rubyiseq.custom<"10">) {
    return
  }
}
