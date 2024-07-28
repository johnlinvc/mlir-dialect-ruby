#ifndef LIB_DIALECT_RUBY_RUBYPATTERNS_TD_
#define LIB_DIALECT_RUBY_RUBYPATTERNS_TD_

include "RubyOps.td"
include "mlir/IR/PatternBase.td"

// def RubyRewriteCallWithAdd : Pat<
//   (Ruby_CallOp $callee, $method, (variadic $arg)),
//   (Ruby_AddOp $callee, $arg)
// >;

#endif  // LIB_DIALECT_RUBY_RUBYPATTERNS_TD_