// RUN: ruby-opt %s | ruby-opt | FileCheck %s

module {
    // CHECK: %{{.*}} = ruby.constant_int "1" : !ruby.int
    %1 = ruby.constant_int "1" : !ruby.int
    // CHECK: %{{.*}} = ruby.add %{{.*}}, %{{.*}} : (!ruby.int, !ruby.int) -> !ruby.int 
    %2 = ruby.add %1,%1 : (!ruby.int, !ruby.int) -> !ruby.int 
}