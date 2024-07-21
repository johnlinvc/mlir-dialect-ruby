// RUN: ruby-translate --mlir-to-ruby %s | FileCheck %s
module {
    // CHECK: 42
    %1 = ruby.constant_int "41" {rb_literal = false} : !ruby.int
    %2 = ruby.constant_int "42" {rb_literal = true} : !ruby.int
}