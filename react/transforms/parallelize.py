import ast
from ast_transforms.utils import *

class Parallelize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        for child in node.body:
            if isinstance(child, ast.For):
                if not hasattr(child, 'is_reduction'):
                    old_range = child.iter
                    child.iter = new_ast_call(
                        new_ast_name('prange'),
                        old_range.args,
                    )

        return node

def transform(node):
    return Parallelize().visit(node)