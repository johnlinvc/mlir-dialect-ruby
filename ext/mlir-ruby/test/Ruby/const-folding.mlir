// RUN: ruby-opt --canonicalize %s | ruby-opt | FileCheck %s

module {
    %0 = ruby.constant_int "3" : !ruby.int
    %1 = ruby.constant_int "4" : !ruby.int
    // CHECK: %{{.*}} = ruby.constant_int "3" : !ruby.int
    %2 = ruby.add %0,%1 : (!ruby.int, !ruby.int) -> !ruby.int 
    %3 = ruby.local_variable_write "foo" = %2 : !ruby.int
   
}
