import ast
from ast_transforms.utils import *

class ConvertTwoSparseOperandStatements(ast.NodeTransformer):
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
        # Create an additional assignment if both operands are sparse
        # Example: 
        # C = spA * spB  => dB = spB * 1; C = spA * dB
        # C = spA + spB  => dB = spB * 1; C = spA + dB
        # C = spA @ spB  => dB = spB * 1; C = spA @ dB
        # Note that "@" is the form of a call "matmul"        
        new_stmts = []
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.left, ast.Name) and isinstance(node.value.right, ast.Name):
                if node.value.left.id in self.sparse_tensors and node.value.right.id in self.sparse_tensors:
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
            if node.value.args[0].id in self.sparse_tensors and node.value.args[1].id in self.sparse_tensors:
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
    return ConvertTwoSparseOperandStatements().visit(node)