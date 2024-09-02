export PATH="/opt/homebrew/opt/llvm@17/bin:$PATH"
#export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
#export export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
cmake -G Ninja ../llvm-project/llvm  -DLLVM_ENABLE_PROJECTS="mlir;lld" -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_BUILD_MLIR_C_DYLIB=1
#cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMLIR_BUILD_MLIR_C_DYLIB=1
