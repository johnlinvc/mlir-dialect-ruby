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
  build_dir = "./ext/mlir-ruby/build"
  cmake_vars = {
    "MLIR_DIR" => "#{prefix}../mlir/lib/cmake/mlir",
    "LLVM_BINARY_DIR" => prefix,
    "LLVM_MAIN_SRC_DIR" => "#{prefix}/../llvm",
    "LLVM_EXTERNAL_LIT" => "#{prefix}/bin/llvm-lit",
    "LLVM_ENABLE_LLD" => "ON"
  }

  desc "clean up build dir"
  task :clean do
    FileUtils.rm_rf build_dir
  end

  desc "configure using cmake"
  task :configure do
    FileUtils.mkdir_p build_dir
    ENV["LDFLAGS"] = "-L/opt/homebrew/opt/llvm/lib"
    ENV["CPPFLAGS"] = "-I/opt/homebrew/opt/llvm/include"
    ENV["PATH"] = "/opt/homebrew/opt/llvm/bin:#{ENV.fetch("PATH", nil)}"
    system("env")
    cmake_vars_str = cmake_vars.map { |k, v| "-D#{k}=#{v}" }.join(" ")
    cmd = "cmake -G 'Unix Makefiles' .. #{cmake_vars_str}}"
    system(cmd, chdir: build_dir)
  end

  desc "build using cmake"
  task :build do
    cmd = "cmake --build ."
    system(cmd, chdir: build_dir)
  end

  desc "test ruby"
  task :testruby do
    cmd = "cmake --build .  --target check-ruby"
    system(cmd, chdir: build_dir)
  end

  desc "test rubyiseq"
  task :testrubyiseq do
    cmd = "cmake --build .  --target check-rubyiseq"
    system(cmd, chdir: build_dir)
  end

  desc "test ruby all"
  task :testrubyall do
    cmd = "cmake --build .  --target check-ruby-all"
    system(cmd, chdir: build_dir)
  end

  task test: %i[testruby testrubyiseq]

end
task dialect: %i[dialect:configure dialect:build]
