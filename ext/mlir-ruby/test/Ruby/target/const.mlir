// RUN: ruby-translate --mlir-to-ruby %s | FileCheck %s
module {
    // CHECK: 40
    %0 = ruby.constant_int "40" {rb_stmt = true} : !ruby.int
    %1 = ruby.constant_int "41" : !ruby.int
    // CHECK-NEXT: 42
    %2 = ruby.constant_int "42" {rb_stmt = true} : !ruby.int
}