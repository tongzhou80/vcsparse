import ast
from .utils import *

class AttachIndexNotation(ast.NodeTransformer):
    def __init__(self):
        self.indices_map = {}

    def visit_FunctionDef(self, node):
        #dump(node.args)
        for arg in node.args.args:
            varname = arg.arg
            indices = []
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                index_str = arg.annotation.args[0].value
                indices = index_str.split(',')
            self.indices_map[varname] = indices
        node.indices = indices
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            assert isinstance(node.left, ast.Name)
            node.indices = self.indices_map[node.left.id]
        elif isinstance(node.op, ast.MatMult):
            assert isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
            a, b = node.left.id, node.right.id
            node.indices = [self.indices_map[a][0], self.indices_map[b][1]]
        else:
            assert False
        return node

    def visit_Assign(self, node):
        assert isinstance(node.targets[0], ast.Name)
        self.generic_visit(node)
        self.indices_map[node.targets[0].id] = node.value.indices
        return node

def transform(tree):
    return AttachIndexNotation().visit(tree)