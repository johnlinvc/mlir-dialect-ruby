// RUN: ruby-opt %s | ruby-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = ruby.foo %{{.*}} : i32
        %res = ruby.foo %0 : i32

        // CHECK: %{{.*}} = ruby.constant_int "1" : !ruby.int
        %1 = ruby.constant_int "1" : !ruby.int

        // Check: %{{.*}} = ruby.local_variable_write "foo" = %1 : ruby.int -> ruby.int
        %2 = ruby.local_variable_write "foo" = %1 : !ruby.int
        // Check: %{{.*}} = ruby.local_variable_read "bar" : ruby.int
        %3 = ruby.local_variable_read "bar" : !ruby.int

        // Check: %{{.*}} = ruby.constant_str "bar" : ruby.string
        %4 = ruby.constant_str "1" : !ruby.string
        
        %5 = ruby.call %1:!ruby.int -> "+"() :  () -> !ruby.int 
        %6 = ruby.call %1:!ruby.int -> "+"(%1) : (!ruby.int) -> !ruby.int 
        %7 = ruby.call %1:!ruby.int -> "+"(%1,%1) : (!ruby.int, !ruby.int) -> !ruby.int 
        %8 = ruby.constant_str "hello world" : !ruby.string
        %9 = ruby.call -> "puts" (%8) : (!ruby.string) -> !ruby.opaque_object

        return
    }
    
    ruby.def "no_arg"():() -> !ruby.string {
    } : !ruby.sym

    ruby.def "hello"(required_args: ["name"]) : (required_args: [!ruby.string]) -> !ruby.string {
    } : !ruby.sym

    %str.0 = ruby.constant_str "hello" : !ruby.string
    ruby.def "str_hello"+(%str.0: !ruby.string)(required_args:["name"]) : (required_args: [!ruby.string]) -> !ruby.string {
    } : !ruby.sym


    // CHECK-LABEL: func @ruby_types(%arg0: !ruby.custom<"10">, %arg1: !ruby.int<"10">)
    func.func @ruby_types(%arg0: !ruby.custom<"10">, %arg1: !ruby.int<"10">) {
        return
    }

    // CHECK-LABEL: func @ruby_string_type(%arg0: !ruby.string<"abc">)
    func.func @ruby_string_type(%arg0: !ruby.string<"abc">) {
        return
    }
}
