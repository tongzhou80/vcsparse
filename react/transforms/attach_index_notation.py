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
            node.indices = []
            for operand in [node.left, node.right]:
                if isinstance(operand, ast.Name):
                    if len(self.indices_map[operand.id]) > len(node.indices):
                        node.indices = self.indices_map[operand.id]
        elif isinstance(node.op, ast.MatMult):
            assert isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
            a, b = node.left.id, node.right.id
            node.indices = [self.indices_map[a][0], self.indices_map[b][1]]
        else:
            assert False
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ('relu', 'exp', 'log', 'neg', 'abs'):
                node.indices = self.indices_map[node.args[0].id]
        return node

    def visit_Assign(self, node):
        assert isinstance(node.targets[0], ast.Name)
        self.generic_visit(node)
        self.indices_map[node.targets[0].id] = node.value.indices
        node.type_comment = 'indices: ' + str(node.value.indices)
        return node

def transform(tree):
    return AttachIndexNotation().visit(tree)