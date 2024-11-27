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



class ConvertToInplaceSpAddForm(ast.NodeTransformer):
    def __init__(self):
        self.sparse_tensors = {}

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            varname = arg.arg
            indices = []
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                index_str = arg.annotation.args[0].value
                indices = index_str.split(',')
                if len(arg.annotation.args) > 1:
                    format = arg.annotation.args[1].value
                    assert format in ('dense', 'csr'), "Only dense and csr format are supported for now!"
                    self.sparse_tensors[varname] = format
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        '''
        Convert a binary operation to a form that uses in-place update.
        Examples:
        C = spA + dB   => C = dB; C = C + spA
        C = spA * dB   => C = 0; C = C + spA * dB
        '''
        new_stmts = []
        if isinstance(node.value, ast.Call):
            if node.value.func.id == 'matmul':
                sparse_args = [arg.id for arg in node.value.args if arg.id in self.sparse_tensors]
                assert len(sparse_args) == 0 or len(sparse_args) == 1
                if len(sparse_args) == 1:
                    sparse_arg = sparse_args[0]
                    node.sparse_info = (sparse_arg, self.sparse_tensors[sparse_arg])
                else:
                    pass
            elif node.value.func.id in ('sum', 'max', 'min'):
                sparse_arg = node.value.args[0].id
                if sparse_arg in self.sparse_tensors:
                    node.sparse_info = (sparse_arg, self.sparse_tensors[sparse_arg])
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.left, ast.Name) and node.value.left.id in self.sparse_tensors:
                # Some sanity check
                assert isinstance(node.value.right, (ast.Name, ast.Constant)), ast.unparse(node)
                if isinstance(node.value.right, ast.Name):
                    assert not node.value.right.id in self.sparse_tensors

                node.sparse_info = (node.value.left.id, self.sparse_tensors[node.value.left.id])
                if isinstance(node.value.op, (ast.Add)):                    
                    new_assign = new_ast_assign(
                        deepcopy_ast_node(node.targets[0], ctx=ast.Store()),
                        deepcopy_ast_node(node.value.right)
                    )
                    new_assign.indices = node.value.left.indices
                    new_stmts.append(new_assign)
                    node.value = new_ast_add(
                        deepcopy_ast_node(node.targets[0], ctx=ast.Load()),
                        node.value.left
                    )
                elif isinstance(node.value.op, (ast.Mult, ast.Div)):
                    if hasattr(node, 'use_sparse_output'):
                        return node
                    new_assign = new_ast_assign(
                        deepcopy_ast_node(node.targets[0], ctx=ast.Store()),
                        new_ast_const(0)
                    )
                    new_assign.indices = node.value.left.indices
                    new_stmts.append(new_assign)
                    node.value = new_ast_add(
                        deepcopy_ast_node(node.targets[0], ctx=ast.Load()),
                        deepcopy_ast_node(node.value)
                    )
        return new_stmts + [node]

def transform(node):
    node = ConvertToInplaceSpAddForm().visit(node)
    return node
