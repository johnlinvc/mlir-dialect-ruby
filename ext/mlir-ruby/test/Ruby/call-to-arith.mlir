// RUN: ruby-opt --ruby-call-to-arith --ruby-type-infer --canonicalize %s | ruby-opt | FileCheck %s

module {
    %0 = ruby.constant_int "3" : !ruby.int
    %1 = ruby.constant_int "4" : !ruby.int
    // CHECK: %[[#int6:]] = ruby.constant_int "6" : !ruby.int
    // CHECK: %[[#int7:]] = ruby.constant_int "7" : !ruby.int
    %2 = ruby.add %0,%1 : (!ruby.int, !ruby.int) -> !ruby.int 
    // CHECK: %{{.*}} = ruby.local_variable_write "foo" = %[[#int7]] : !ruby.int
    %3 = ruby.local_variable_write "foo" = %2 : !ruby.int
    %5 = ruby.call %0:!ruby.int -> "+"(%0) : (!ruby.int) -> !ruby.opaque_object 
    // CHECK-NEXT: %{{.*}} = ruby.local_variable_write "bar" = %[[#int6]] : !ruby.int
    %6 = ruby.local_variable_write "bar" = %5 : !ruby.opaque_object 
}
