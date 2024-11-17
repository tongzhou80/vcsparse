import ast
from .utils import *

class AttachIndexNotation(ast.NodeTransformer):
    def __init__(self):
        self.indices_map = {}
        self.index_range = {}
        self.tensor_format = {}

    def visit_FunctionDef(self, node):
        #dump(node.args)
        for arg in node.args.args:
            varname = arg.arg
            indices = []
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                index_str = arg.annotation.args[0].value
                indices = index_str.split(',')
                self.tensor_format[varname] = 'dense'
                if len(arg.annotation.args) > 1:
                    self.tensor_format[varname] = arg.annotation.args[1].value

            self.indices_map[varname] = indices
            for pos,index in enumerate(indices):
                if index not in self.index_range:
                    #self.index_range[index] = f'{varname}.shape[{pos}]'
                    self.index_range[index] = (varname, pos)
        node.indices_map = self.indices_map
        node.index_range = self.index_range
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

    def visit_Compare(self, node):
        node.indices = []
        for operand in node.left, node.comparators:
            if isinstance(operand, ast.Name):
                if len(self.indices_map[operand.id]) > len(node.indices):
                    node.indices = self.indices_map[operand.id]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ('relu', 'exp', 'log', 'neg', 'abs', 'where'):
                node.indices = self.indices_map[node.args[0].id]
            elif node.func.id in ['sum', 'max', 'min']:
                full_indices = self.indices_map[node.args[0].id]
                axis = node.args[1].value
                node.indices = [full_indices[i] for i in range(len(full_indices)) if i != axis]
            elif node.func.id == 'matmul':
                node.indices = [self.indices_map[node.args[0].id][0], self.indices_map[node.args[1].id][1]]
        return node

    def visit_Name(self, node):
        if node.id in self.indices_map:
            node.indices = self.indices_map[node.id]
        return node

    def visit_Assign(self, node):
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        self.generic_visit(node)
        self.indices_map[target.id] = node.value.indices
        target.indices = node.value.indices
        node.type_comment = 'indices: ' + str(node.value.indices)
        return node

def transform(tree):
    return AttachIndexNotation().visit(tree)