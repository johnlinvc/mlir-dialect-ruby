// RUN: rubyiseq-opt %s | rubyiseq-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = rubyiseq.foo %{{.*}} : i32
        %res = rubyiseq.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @rubyiseq_types(%arg0: !rubyiseq.custom<"10">)
    func.func @rubyiseq_types(%arg0: !rubyiseq.custom<"10">) {
        return
    }
}
