import ast
from ast_transforms.utils import *

class Parallelize(ast.NodeTransformer):
    def visit_For(self, node):
        if not hasattr(node, 'is_reduction'):
            old_range = node.iter
            node.iter = new_ast_call(
                        new_ast_attribute(new_ast_name('numba'), 'prange'),
                        old_range.args,
                    )
        return node

def transform(node):
    return Parallelize().visit(node)