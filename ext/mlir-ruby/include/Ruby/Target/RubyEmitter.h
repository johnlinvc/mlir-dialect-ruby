#ifndef MLIR_RUBY_TARGET_RUBY_EMITTER_H
#define MLIR_RUBY_TARGET_RUBY_EMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ruby {
    LogicalResult translateToRuby(Operation *op, raw_ostream &os);
} // namespace ruby
} // namespace mlir


#endif // MLIR_RUBY_TARGET_RUBY_EMITTER_H