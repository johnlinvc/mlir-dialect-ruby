# frozen_string_literal: true

require "bundler/gem_tasks"
require "minitest/test_task"
require 'fileutils'

Minitest::TestTask.create

require "rubocop/rake_task"

RuboCop::RakeTask.new

task default: %i[test rubocop]

namespace :dialect do

  prefix = "/Users/johnlinvc/projs/ruby-mlir/llvm-project/build"
  build_dir = "./ext/standalone/build"

  desc "configure using cmake"
  task :configure do
    FileUtils.mkdir_p build_dir
    cmd = "cmake -G Ninja .. -DMLIR_DIR=#{prefix}/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=#{prefix}/bin/llvm-lit"
    system(cmd, chdir: build_dir)
  end

  desc "build using cmake"
  task :build do
    cmd = "cmake --build . --target check-standalone"
    system(cmd, chdir: build_dir)
  end

end
task dialect: %i[dialect:configure dialect:build]
