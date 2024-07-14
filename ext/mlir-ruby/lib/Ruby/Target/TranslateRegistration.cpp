#include "mlir/Tools/mlir-translate/Translation.h"
#include "Ruby/Target/RubyEmitter.h"
#include "Ruby/RubyDialect.h"

using namespace mlir;
void registerToRubyTranslation(){
      TranslateFromMLIRRegistration reg(
      "mlir-to-cpp", "translate from mlir to cpp",
      [](Operation *op, raw_ostream &output) {
        return ruby::translateToRuby(
            op, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<ruby::RubyDialect>();
        // clang-format on
      });
}
