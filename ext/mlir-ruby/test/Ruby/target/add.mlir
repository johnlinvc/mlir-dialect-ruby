// RUN: ruby-translate --mlir-to-ruby %s | FileCheck %s
module {
    %0 = ruby.constant_int "3" : !ruby.int
    %1 = ruby.constant_int "4" : !ruby.int
    %2 = ruby.add %1,%0 : (!ruby.int, !ruby.int) -> !ruby.int 
    // CHECK: 3 + 4
    %3 = ruby.add %0,%1 {rb_stmt = true} : (!ruby.int, !ruby.int) -> !ruby.int 
}