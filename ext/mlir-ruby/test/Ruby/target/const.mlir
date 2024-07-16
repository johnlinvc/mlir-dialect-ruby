// RUN: ruby-translate --mlir-to-ruby %s | FileCheck %s
module {
    // CHECK: 42
    %1 = ruby.constant_int "42" : !ruby.int
}