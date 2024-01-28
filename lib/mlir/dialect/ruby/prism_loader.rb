# frozen_string_literal: true

require "prism"
require "mlir"

module MLIR
  module Dialect
    module Ruby
      # visit prism ast
      class PrismVisitor
        attr_reader :context, :stmts

        def initialize(context = nil)
          @context = context || MLIR::CAPI.mlirContextCreate
          @ssa_counter = 0
          @stmts = []
          MLIR::CAPI.register_all_upstream_dialects(@context)
          MLIR::CAPI.mlirDialectHandleRegisterDialect(MLIR::Dialect::Ruby::CAPI.mlirGetDialectHandle__ruby__, @context)
        end

        def visit_program(node)
          visit_statements(node.statements)
        end

        def visit_statements(node)
          ret = []
          node.body.each do |stmt|
            # pp stmt
            ret << visit(stmt)
          end
          ret[-1]
        end

        def visit_call(node)
          receiver = visit(node.receiver)
          name = node.name
          args = visit_arguments(node.arguments)
          build_call_stmt(receiver, name, args)
        end

        def visit_arguments(node)
          node.arguments.map do |arg|
            visit(arg)
          end
        end

        def visit_parentheses(node)
          visit(node.body)
        end

        def visit_integer(node)
          build_int_stmt(node.value)
        end

        def visit_local_variable_write(node)
          value = visit(node.value)
          build_local_variable_write_stmt(node.name, value)
        end

        def visit_local_variable_read(node)
          build_local_variable_read_stmt(node.name)
        end

        def visit(node)
          type = node.type.to_s
          method_name = "visit_#{type.split("_")[..-2].join("_")}"
          raise "not implemented: #{method_name}" unless respond_to?(method_name)

          send(method_name, node)
        end

        def with_new_ssa_var
          ret = "%#{@ssa_counter}"
          yield ret if block_given?
          @ssa_counter += 1
          ret
        end

        def build_call_stmt(receiver, name, args)
          with_new_ssa_var do |ssa_var|
            @stmts << "  #{ssa_var} = ruby.call #{receiver}, \"#{name}\", #{args.join(", ")}"
          end
        end

        def build_local_variable_write_stmt(name, value)
          with_new_ssa_var do |ssa_var|
            @stmts << "  #{ssa_var} = ruby.local_variable_write \"#{name}\" = #{value} : !ruby.int -> !ruby.int"
          end
        end

        def build_local_variable_read_stmt(name)
          with_new_ssa_var do |ssa_var|
            @stmts << "  #{ssa_var} = ruby.local_variable_read \"#{name}\" : !ruby.int"
          end
        end

        def build_int_stmt(value)
          # MLIR::CAPI.mlirBuildIntLit(@context, MLIR::CAPI.mlirIntegerTypeGet(@context, 64), value)
          with_new_ssa_var do |ssa_var|
            @stmts << "  #{ssa_var} = ruby.constant_int <\"#{value}\"> : !ruby.int"
          end
        end
      end

      # convert ruby code to mlir via prism
      class PrismLoader
        attr_reader :prog, :ast, :visitor

        def initialize(program)
          @prog = program
          @ast = Prism.parse(@prog)
          @visitor = PrismVisitor.new
        end

        def to_module
          @visitor.visit(@ast.value)
          # pp @ast.value
          # puts @visitor.stmts
          @visitor.stmts
        end
      end
    end
  end
end
