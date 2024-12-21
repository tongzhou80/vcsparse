from ast_transforms.utils import *
import ast_comments as ast

class GenAPPyCode(ast.NodeTransformer):
    def __init__(self, *args):
        pass

    def visit_FunctionDef(self, node):
        # Starting with _ means a kernel function
        if node.name.startswith('_'):
            node.decorator_list.append(new_ast_node_from_str(f"appy.jit"))
            self.generic_visit(node)    
        return node

    def visit_For(self, node):
        if not hasattr(node, 'is_reduction'):
            comment = ast.Comment(value='#pragma parallel for', inline=False)
            return comment, node
        return node

def transform(tree, *args):
    return GenAPPyCode(*args).visit(tree)