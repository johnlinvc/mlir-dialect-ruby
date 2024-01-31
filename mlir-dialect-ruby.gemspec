# frozen_string_literal: true

require_relative "lib/mlir/dialect/ruby/version"

Gem::Specification.new do |spec|
  spec.name = "mlir-dialect-ruby"
  spec.version = MLIR::Dialect::Ruby::VERSION
  spec.authors = ["johnlinvc"]
  spec.email = ["johnlinvc@gmail.com"]

  spec.summary = "A MLIR dialect for ruby"
  spec.description = "A MLIR dialect for ruby"
  spec.homepage = "https://github.com/johnlinvc/mlir-dialect-ruby"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.3.0"

  # spec.metadata["allowed_push_host"] = "TODO: Set to your gem server 'https://example.com'"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/johnlinvc/mlir-dialect-ruby"
  spec.metadata["changelog_uri"] = "https://github.com/johnlinvc/mlir-dialect-ruby/blob/main/CHANGELOG.md"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) ||
        f.start_with?(*%w[bin/ test/ spec/ features/ .git .github appveyor Gemfile])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # Uncomment to register a new dependency of your gem
  spec.add_dependency "ffi", "~> 1.16"
  spec.add_dependency "mlir", "~> 0.1.1"
  spec.add_dependency "prism", "~> 0.19"

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
  spec.metadata["rubygems_mfa_required"] = "true"
end
