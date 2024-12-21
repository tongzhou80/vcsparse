import ast
from ast_transforms.utils import *

class GenNumbaCode(ast.NodeTransformer):
    def __init__(self, parallel=False):
        if parallel:
            self.parallel = True
        else:
            self.parallel = False

    def visit_FunctionDef(self, node):
        # Starting with _ means a kernel function
        if node.name.startswith('_'):
            node.decorator_list.append(new_ast_node_from_str(f"numba.njit(parallel={self.parallel})"))        
        return node

def transform(tree, *args):
    return GenNumbaCode(*args).visit(tree)