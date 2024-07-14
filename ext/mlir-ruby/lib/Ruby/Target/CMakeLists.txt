add_mlir_translation_library(MLIRTargetRuby
  TranslateRegistration.cpp
  TranslateToRuby.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Ruby/Target

  LINK_LIBS PUBLIC
  MLIRRuby
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  )
