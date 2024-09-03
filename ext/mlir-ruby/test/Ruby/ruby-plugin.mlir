// RUN: mlir-opt %s --load-dialect-plugin=%ruby_libs/RubyPlugin%shlibext --pass-pipeline="builtin.module()" | FileCheck %s

// CHECK: module {
module {

}
