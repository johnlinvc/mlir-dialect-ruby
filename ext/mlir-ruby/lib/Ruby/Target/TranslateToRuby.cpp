#include "Ruby/RubyDialect.h"
#include "Ruby/RubyOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ruby;

namespace {
  struct RubyEmitter
  {
    explicit RubyEmitter(raw_ostream &os);
    LogicalResult emitOperation(Operation &op, bool skipStmtCheck=false);
    LogicalResult emitOperand(Value operand);

    raw_indented_ostream &ostream() { return os; };
  private:
    /// Output stream to emit to.
    raw_indented_ostream os;
  };
}

static LogicalResult printOperation(RubyEmitter &emitter, ruby::LocalVariableWriteOp op, bool skipStmtCheck = false) {
  if (!op->getAttrDictionary().get("rb_stmt")) {
    if (!skipStmtCheck) {
      return success();
    }
  }
  Attribute value = op.getVarNameAttr();
  raw_ostream &os = emitter.ostream();
  if(!isa<StringAttr>(value)) {
    return failure();
  }
  if (auto sAttr = dyn_cast<StringAttr>(value)) {
    os << sAttr.getValue().str();
  }
  os << " = ";
  if( failed(emitter.emitOperand(op.getInput())) )
    return failure();
  if (op->getAttrDictionary().get("rb_stmt")) {
    os << "\n";
  }
  return success();
}

static LogicalResult printOperation(RubyEmitter &emitter, ruby::AddOp op, bool skipStmtCheck = false) {
  if (!op->getAttrDictionary().get("rb_stmt")) {
    if (!skipStmtCheck) {
      return success();
    }
  }
  if( failed(emitter.emitOperand(op.getLhs())) )
    return failure();
  raw_ostream &os = emitter.ostream();
  os << " + ";
  if( failed(emitter.emitOperand(op.getRhs())) )
    return failure();
  if (op->getAttrDictionary().get("rb_stmt")) {
    os << "\n";
  }
  return success();
}

static LogicalResult printOperation(RubyEmitter &emitter, ruby::ConstantIntOp op, bool skipStmtCheck = false) {
  if (!op->getAttrDictionary().get("rb_stmt")) {
    if (!skipStmtCheck) {
      return success();
    }
  }
  Attribute value = op.getInputAttr();
  raw_ostream &os = emitter.ostream();
  if(!isa<StringAttr>(value)) {
    return failure();
  }
  if (auto sAttr = dyn_cast<StringAttr>(value)) {
    os << sAttr.getValue().str();
  }
  if (op->getAttrDictionary().get("rb_stmt")) {
    os << "\n";
  }
  return success();
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

LogicalResult RubyEmitter::emitOperand(Value operand) {
  auto status = emitOperation(*operand.getDefiningOp(), /*skipStmtCheck=*/true);
  if (failed(status))
    return failure();
  return success();
}

LogicalResult RubyEmitter::emitOperation(Operation &op, bool skipStmtCheck/*=false*/) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<ModuleOp>([&](auto op)
                          { return printOperation(*this, op); })
          .Case<ruby::ConstantIntOp>([&](auto op)
                                  { return printOperation(*this, op, skipStmtCheck); })
          .Case<ruby::AddOp>([&](auto op)
                                  { return printOperation(*this, op, skipStmtCheck); })
          .Case<ruby::LocalVariableWriteOp>([&](auto op)
                                  { return printOperation(*this, op, skipStmtCheck); })
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
