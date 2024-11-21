import ast
from ast_transforms.utils import *

class ToSparseInplaceAddForm(ast.NodeTransformer):
    '''
    This pass does a number of transformations:
    1. It checks binary operators, and if both operands are sparse, it 
    creates a dense copy for one of them, and adds the other operand to it.
    After the conversion, each statement will have at most one sparse operand.
    matmul is a special binary operator, and we will always create a dense copy
    for the second operand (if both operands are CSR format), because CSR x dense
    is easier to do. Note that non-CSR formats are not supported yet.
    2. For "dA = spA", "dA = spA * B", "dA = spA @ B" assignments, it converts them to an in-place update form,
    like "dA = 0; dA = dA + spA". The in-place sparse update statement will be 
    marked as having sparse operand spA, which will later be processed by the
    sparsify_loop pass.
    For example, for assignment "C = spA * spB", the conversion process is
    C = spA * dB   => C = 0; C = C + spA * dB

    C = dA * spB   => C = spB * dA
                   => C = 0; C = C + spB * dA

    C = spA * spB  => dB = spB * 1; C = spA * dB
                   => dB = 0; dB = dB + spB * 1; C = 0; C = C + spA * dB

    C = spA + dB   => C = dB; C = C + spA

    C = dA + spB   => C = spB + dA
                   => C = dA; C = C + spB

    C = spA + spB  => dB = spB * 1; C = spA + dB
                   => dB = 0; dB = dB + spB * 1; C = dB; C = C + spA * 1

    C = spA @ dB   => C = spA @ dB
                   => it needs to be converted to form C = 0; C = C + spA @ dB
                   => but this should be handled by the fix-reduction pass
                   => from the sparse-dense assignment point of view, C = 0 is not needed

    C = spA @ spB  => dB = spB * 1; C = spA @ dB
                   => dB = 0; dB = dB + spB * 1; C = spA @ dB

    C = dA @ spB   => C = dA @ spB
                   => no conversion seems necessary, just applying sparse-loop pass should work
    '''
    pass

class ConvertTwoSparseOperandStatements(ast.NodeTransformer):
    def __init__(self):
        self.tensor_format = {}

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            varname = arg.arg
            indices = []
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                index_str = arg.annotation.args[0].value
                indices = index_str.split(',')
                self.tensor_format[varname] = 'dense'
                if len(arg.annotation.args) > 1:
                    format = arg.annotation.args[1].value
                    assert format in ('dense', 'csr'), "Only dense and csr format are supported for now!"
                    self.tensor_format[varname] = format
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # Create an additional assignment if both operands are sparse
        # Example: 
        # C = spA * spB  => dB = spB * 1; C = spA * dB
        # C = spA + spB  => dB = spB * 1; C = spA + dB
        # C = spA @ spB  => dB = spB * 1; C = spA @ dB
        # Note that "@" is the form of a call "matmul"
        formats = self.tensor_format
        new_stmts = []
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.left, ast.Name) and isinstance(node.value.right, ast.Name):
                if formats[node.value.left.id] == 'csr' and formats[node.value.right.id] == 'csr':
                    new_var = '__d' + node.value.right.id
                    new_stmt = new_ast_assign(
                        new_ast_name(new_var, ctx=ast.Store()),
                        new_ast_mul(
                            new_ast_name(node.value.right.id),
                            new_ast_const(1)
                        )
                        
                    )
                    new_stmts.append(new_stmt)
                    node.value.right = new_ast_name(new_var)
        elif isinstance(node.value, ast.Call) and node.value.func.id == 'matmul':
            if formats[node.value.args[0].id] == 'csr' and formats[node.value.args[1].id] == 'csr':
                new_var = '__d' + node.value.args[1].id
                new_stmt = new_ast_assign(
                    new_ast_name(new_var, ctx=ast.Store()),
                    new_ast_mul(
                        new_ast_name(node.value.args[1].id),
                        new_ast_const(1)
                    )
                )
                new_stmts.append(new_stmt)
                node.value.args[1] = new_ast_name(new_var)
        return new_stmts + [node]

def transform(node):
    node = ConvertTwoSparseOperandStatements().visit(node)
    return node
