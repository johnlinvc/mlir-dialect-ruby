#include "Ruby/RubyDialect.h"
#include "Ruby/RubyOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ruby;

namespace {
  struct RubyEmitter
  {
    explicit RubyEmitter(raw_ostream &os);
    LogicalResult emitOperation(Operation &op);

    raw_indented_ostream &ostream() { return os; };
  private:
    /// Output stream to emit to.
    raw_indented_ostream os;
  };
}

static LogicalResult printOperation(RubyEmitter &emitter, ruby::ConstantIntOp op) {
  if (!op->getAttrDictionary().get("rb_stmt")) {
    return success();
  }
  Attribute value = op.getInputAttr();
  raw_ostream &os = emitter.ostream();
  if (auto sAttr = dyn_cast<StringAttr>(value)) {
    os << sAttr.getValue().str();
    return success();
  }
  return failure();
}

static LogicalResult printOperation(RubyEmitter &emitter, ModuleOp moduleOp) {
  // RubyEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

RubyEmitter::RubyEmitter(raw_ostream &os)
    : os(os){
}

LogicalResult RubyEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<ModuleOp>([&](auto op)
                          { return printOperation(*this, op); })
          .Case<ruby::ConstantIntOp>([&](auto op)
                                  { return printOperation(*this, op); })
          // .Case<ruby::AddOp>([&](auto op)
          //                         { return printOperation(*this, op); })
          .Default([&](Operation *)
                   { return op.emitOpError("unable to find printer for op"); });
  if (failed(status))
    return failure();
  return success();
}

namespace mlir {
  namespace ruby {
    LogicalResult translateToRuby(Operation *op, raw_ostream &output)
    {
      // Translate the operation to Ruby.
      RubyEmitter emitter(output);
      return emitter.emitOperation(*op);
    }
  } // namespace ruby
} // namespace mlir
