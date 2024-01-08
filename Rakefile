# frozen_string_literal: true

require "bundler/gem_tasks"
require "minitest/test_task"
require "fileutils"

Minitest::TestTask.create

require "rubocop/rake_task"

RuboCop::RakeTask.new

task default: %i[test rubocop]

namespace :dialect do
  prefix = "/Users/johnlinvc/projs/ruby-mlir/llvm-project/build"
  build_dir = "./ext/standalone/build"

  desc "clean up build dir"
  task :clean do
    FileUtils.rm_rf build_dir
  end

  desc "configure using cmake"
  task :configure do
    FileUtils.mkdir_p build_dir
    ENV["LDFLAGS"]="-L/opt/homebrew/opt/llvm/lib"
    ENV["CPPFLAGS"]="-I/opt/homebrew/opt/llvm/include"
    ENV["PATH"]="/opt/homebrew/opt/llvm/bin:#{ENV["PATH"]}"
    system("env")
    cmd = "cmake -G 'Unix Makefiles' .. -DMLIR_DIR=#{prefix}../mlir/lib/cmake/mlir -DLLVM_BINARY_DIR=#{prefix} -DLLVM_MAIN_SRC_DIR=#{prefix}/../llvm -DLLVM_EXTERNAL_LIT=#{prefix}/bin/llvm-lit -DLLVM_ENABLE_LLD=ON"
    system(cmd, chdir: build_dir)
  end

  desc "build using cmake"
  task :build do
    cmd = "cmake --build . --target check-ruby"
    system(cmd, chdir: build_dir)
  end
end
task dialect: %i[dialect:configure dialect:build]
