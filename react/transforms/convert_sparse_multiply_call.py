import ast
from ast_transforms.utils import *

class ConvertSparseMultiplyFunc(ast.NodeTransformer):
    '''
    This converts A.multiply(B) to A * B, where A is sparse.
    '''
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'multiply':
            node = new_ast_mul(node.func.value, node.args[0])
        return node

def transform(node):
    return ConvertSparseMultiplyFunc().visit(node)