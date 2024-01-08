// RUN: ruby-opt %s | ruby-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = ruby.foo %{{.*}} : i32
        %res = ruby.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @ruby_types(%arg0: !ruby.custom<"10">, %arg1: !ruby.int<"10">)
    func.func @ruby_types(%arg0: !ruby.custom<"10">, %arg1: !ruby.int<"10">) {
        return
    }
}
