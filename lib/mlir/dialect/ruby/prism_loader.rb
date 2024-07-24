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
        attr_reader :context, :stmts, :attr_queue

        def initialize(context = nil)
          @context = context || MLIR::CAPI.mlirContextCreate
          @ssa_counter = 0
          @ssa_prefixes = []
          @stmts = []
          @attr_queue = []
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
            attr_queue << { "rb_stmt" => true }
            ret = visit(stmt)
          end
          ret
        end

        def visit_call(node)
          receiver = nil
          if node.receiver
            attr_queue << {}
            receiver = visit(node.receiver)
          end
          name = node.name
          args = visit_arguments(node.arguments)
          build_call_stmt(receiver, name, args)
        end

        def visit_arguments(node)
          node.arguments.map do |arg|
            attr_queue << {}
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

        # Utility methods

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

        def pop_gen_attr_dict
          attr_dict = attr_queue.pop
          return "" unless attr_dict

          res = "{"
          body_pairs = attr_dict.each.map do |key, value|
            "#{key} = #{value.inspect}"
          end
          res += body_pairs.join(", ")
          res += "}"
          res
        end

        def with_new_ssa_var
          ret = "%#{ssa_prefix}#{@ssa_counter}"
          raise "must have a block" unless block_given?
          attr_dict = pop_gen_attr_dict
          type = yield ret,attr_dict
          @ssa_counter += 1
          SSARetValue.new(ret, type)
        end

        # Build MLIR statements

        # TODO: Use MLIR for this specialization instead
        PLUS_STMT_TPL_STR = <<~PLUS_STMT_TPL.strip
          <%= ssa_var %> = ruby.add <%= lhs.ssa_var %>\
          , <%= rhs.ssa_var %> <%= attr_dict %> :\
          (!ruby.int, !ruby.int) -> !ruby.int
        PLUS_STMT_TPL
        PLUS_STMT_TPL = ERB.new(PLUS_STMT_TPL_STR)
        def build_plus_stmt(lhs, rhs)
          with_new_ssa_var do |ssa_var, attr_dict|
            @stmts << PLUS_STMT_TPL.result(binding)
            ret_type = "!ruby.int"
            ret_type
          end
        end

        CALL_STMT_TPL_STR = <<~CALL_STMT_TPL.strip
          <%= ssa_var %> = ruby.call <%= receiver_info %>\
          -> "<%= name %>"(<%= args_ssa_values %>) \
          : (<%= arg_types %>) -> <%= ret_type %>
        CALL_STMT_TPL
        CALL_STMT_TPL = ERB.new(CALL_STMT_TPL_STR)

        def build_call_stmt(receiver, name, args)
          plus_optimize = name == :'+' && receiver && args.size == 1
          if plus_optimize
            return build_plus_stmt(receiver, args[0])
          end
          with_new_ssa_var do |ssa_var|
            receiver_info = receiver ? "#{receiver.ssa_var} : #{receiver.type} " : ""
            args_ssa_values = args.map(&:ssa_var).join(", ")
            arg_types = args.map(&:type).join(", ")
            ret_type = "!ruby.opaque_object"
            @stmts << CALL_STMT_TPL.result(binding)
            ret_type
          end
        end

        LOCAL_VAR_WRITE_TPL_STR = <<~LOCAL_VAR_WRITE_TPL.strip
          <%= ssa_var %> = ruby.local_variable_write "<%= name %>"\
          = <%= value.ssa_var %> <%= attr_dict %> : <%= value.type %>
        LOCAL_VAR_WRITE_TPL
        LOCAL_VAR_WRITE_TPL = ERB.new(LOCAL_VAR_WRITE_TPL_STR)
        def build_local_variable_write_stmt(name, value)
          with_new_ssa_var do |ssa_var, attr_dict|
            @stmts << LOCAL_VAR_WRITE_TPL.result(binding)
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
          with_new_ssa_var do |ssa_var, attr_dict|
            ret_type = "!ruby.int"
            @stmts << "  #{ssa_var} = ruby.constant_int \"#{value}\" #{attr_dict} : #{ret_type}"
            ret_type
          end
        end

        def build_string_stmt(value)
          with_new_ssa_var do |ssa_var|
            ret_type = "!ruby.string"
            @stmts << "  #{ssa_var} = ruby.constant_str \"#{value}\" #{pop_gen_attr_dict} : #{ret_type}"
            ret_type
          end
        end

        def build_def_stmt_params(parameters)
          params = "("
          if parameters[:requireds]
            requireds = parameters[:requireds].map do |arg|
              "\"#{arg}\""
            end.join(",")
            params += "required_args: [#{requireds}]"
          end
          params += ")"
          params
        end

        def build_def_stmt_param_types(parameters)
          param_types = "("
          if parameters[:requireds]
            param_types += "required_args: [#{parameters[:requireds].map do
                                                "!ruby.opaque_object"
                                              end.join(",")}]"
          end
          param_types += ")"
          param_types
        end

        DEF_STMT_TPL_STR = <<~DEF_STMT_TPL.strip
          <%= def_var %> = ruby.def "<%= name %>"<%= receiver_part %> <%= params_part %> \
          : \
          <%= param_types_part %> -> !ruby.opaque_object
        DEF_STMT_TPL
        DEF_STMT_TPL = ERB.new(DEF_STMT_TPL_STR)

        def build_def_stmt_front_part(name, receiver, parameters, def_var)
          receiver_part = receiver ? "+(#{receiver.ssa_var} : #{receiver.type})" : ""
          params_part = build_def_stmt_params(parameters)
          param_types_part = build_def_stmt_param_types(parameters)
          stmt = DEF_STMT_TPL.result(binding)
          @stmts << stmt
        end

        def build_def_stmt_region_part
          @stmts << "{"
          yield
          ret_type = "!ruby.sym"
          @stmts << "} : #{ret_type}"
          ret_type
        end

        def build_def_stmt(name, receiver, parameters)
          with_new_ssa_var do |def_var|
            build_def_stmt_front_part(name, receiver, parameters, def_var)
            build_def_stmt_region_part do
              with_ssa_prefix(name) do
                value = yield
                stmts << "  ruby.return #{value.ssa_var} : #{value.type}" if stmts.last !~ /\A\s*ruby.return/
              end
            end
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
        MODULE_TPL = <<~ERB
          module {
          <%= stmts.join("\n") %>
          }
        ERB
        def module_from_stmts(stmts)
          ERB.new(MODULE_TPL).result(binding)
        end

        def to_module
          @visitor.visit(@ast.value)
          # pp @ast.value
          # puts @visitor.stmts
          stmts = @visitor.stmts
          module_from_stmts(stmts)
        end
      end
    end
  end
end
