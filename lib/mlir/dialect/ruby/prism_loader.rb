# frozen_string_literal: true

require "prism"
require "mlir"
require "erb"

module MLIR
  module Dialect
    module Ruby
      # A struct to hold SSA variable and its type
      SSARetValue = Struct.new(:ssa_var, :type)

      # rubocop:disable Metrics/ClassLength
      # visit prism ast
      class PrismVisitor
        attr_reader :context, :stmts

        def initialize(context = nil)
          @context = context || MLIR::CAPI.mlirContextCreate
          @ssa_counter = 0
          @ssa_prefixes = []
          @stmts = []
          MLIR::CAPI.register_all_upstream_dialects(@context)
          MLIR::CAPI.mlirDialectHandleRegisterDialect(MLIR::Dialect::Ruby::CAPI.mlirGetDialectHandle__ruby__, @context)
        end

        def visit_program(node)
          visit_statements(node.statements)
        end

        def visit_statements(node)
          ret = nil
          node.body.each do |stmt|
            # pp stmt
            ret = visit(stmt)
          end
          ret
        end

        def visit_call(node)
          receiver = node.receiver ? visit(node.receiver) : nil
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

        def visit_string(node)
          build_string_stmt(node.unescaped)
        end

        def visit_parameters(node)
          {
            requireds: node.requireds&.map(&:name)
          }
        end

        def visit_def(node)
          receiver = node.receiver ? visit(node.receiver) : nil
          parameters = visit_parameters(node.parameters)
          build_def_stmt(node.name, receiver, parameters) do
            visit_statements(node.body)
          end
        end

        def visit(node)
          type = node.type.to_s
          method_name = "visit_#{type.split("_")[..-2].join("_")}"
          raise "not implemented: #{method_name}" unless respond_to?(method_name)

          send(method_name, node)
        end

        def ssa_prefix
          @ssa_prefixes.map { "#{_1}." }.join
        end

        def with_ssa_prefix(prefix)
          raise "must have a block" unless block_given?

          @ssa_prefixes << prefix
          currnt_counter = @ssa_counter
          yield
          @ssa_counter = currnt_counter
          @ssa_prefixes.pop
        end

        def with_new_ssa_var
          ret = "%#{ssa_prefix}#{@ssa_counter}"
          raise "must have a block" unless block_given?

          type = yield ret
          @ssa_counter += 1
          SSARetValue.new(ret, type)
        end

        CALL_STMT_TPL_STR = <<~CALL_STMT_TPL.strip
          <%= ssa_var %> = ruby.call <%= receiver_info %>\
          -> "<%= name %>"(<%= args_ssa_values %>) \
          : (<%= arg_types %>) -> <%= ret_type %>
        CALL_STMT_TPL
        CALL_STMT_TPL = ERB.new(CALL_STMT_TPL_STR)

        def build_call_stmt(receiver, name, args)
          with_new_ssa_var do |ssa_var|
            receiver_info = receiver ? "#{receiver.ssa_var} : #{receiver.type} " : ""
            args_ssa_values = args.map(&:ssa_var).join(", ")
            arg_types = args.map(&:type).join(", ")
            ret_type = "!ruby.opaque_object"
            @stmts << CALL_STMT_TPL.result(binding)
            ret_type
          end
        end

        def build_local_variable_write_stmt(name, value)
          with_new_ssa_var do |ssa_var|
            @stmts << "  #{ssa_var} = ruby.local_variable_write \"#{name}\" = #{value.ssa_var} : #{value.type} "
            value.type
          end
        end

        def build_local_variable_read_stmt(name)
          with_new_ssa_var do |ssa_var|
            ret_type = "!ruby.opaque_object"
            @stmts << "  #{ssa_var} = ruby.local_variable_read \"#{name}\" : #{ret_type}"
            ret_type
          end
        end

        def build_int_stmt(value)
          # MLIR::CAPI.mlirBuildIntLit(@context, MLIR::CAPI.mlirIntegerTypeGet(@context, 64), value)
          with_new_ssa_var do |ssa_var|
            ret_type = "!ruby.int"
            @stmts << "  #{ssa_var} = ruby.constant_int \"#{value}\" : #{ret_type}"
            ret_type
          end
        end

        def build_string_stmt(value)
          with_new_ssa_var do |ssa_var|
            ret_type = "!ruby.string"
            @stmts << "  #{ssa_var} = ruby.constant_str \"#{value}\" : #{ret_type}"
            ret_type
          end
        end

        def build_def_stmt(name, receiver, parameters)
          with_new_ssa_var do |def_var|
            stmt = "#{def_var} = ruby.def \"#{name}\""
            stmt += "+(#{receiver.ssa_var} : #{receiver.type})" if receiver
            params = "("
            if parameters[:requireds]
              params += "required_args: [#{parameters[:requireds].map do
                                             "\"#{_1}\""
                                           end.join(",")}]"
            end
            params += ")"
            stmt += params
            stmt += ":"
            param_types = "("
            if parameters[:requireds]
              param_types += "required_args: [#{parameters[:requireds].map do
                                                  "!ruby.opaque_object"
                                                end.join(",")}]"
            end
            param_types += ")"
            stmt += param_types
            stmt += " -> !ruby.opaque_object"
            @stmts << stmt
            @stmts << "{"
            with_ssa_prefix("name") do
              value = yield
              stmts << "  ruby.return #{value.ssa_var} : #{value.type}" if stmts.last !~ /\A\s*ruby.return/
            end
            ret_type = "!ruby.sym"
            @stmts << "} : #{ret_type}"
            ret_type
          end
        end

        # rubocop:enable Metrics/ClassLength
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
