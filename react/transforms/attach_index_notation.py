import ast
from .utils import *

class AttachIndexNotation(ast.NodeTransformer):
    def __init__(self):
        self.notations = {}

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            varname = arg.arg
            index_str = arg.annotation.args[0].value
            indices = index_str.split(',')
            self.notations[varname] = indices
        node.index_notations = self.notations
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            assert isinstance(node.left, ast.Name)
            node.index_notations = self.notations[node.left.id]
        elif isinstance(node.op, ast.MatMult):
            assert isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
            a, b = node.left.id, node.right.id
            node.index_notations = [self.notations[a][0], self.notations[b][1]]
        else:
            assert False
        return node

    def visit_Assign(self, node):
        dump(node)
        assert isinstance(node.targets[0], ast.Name)
        self.generic_visit(node.value)
        self.notations[node.targets[0].id] = node.value.index_notations
        return node

def transform(tree):
    return AttachIndexNotation().visit(tree)