import ast
from ast_transforms.utils import *

class AttachIndexNotation(ast.NodeTransformer):
    def __init__(self):
        self.indices_map = {}
        self.index_range = {}
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
            self.indices_map[varname] = indices
            for pos,index in enumerate(indices):
                if index not in self.index_range:
                    #self.index_range[index] = f'{varname}.shape[{pos}]'
                    self.index_range[index] = (varname, pos)
        node.indices_map = self.indices_map
        node.index_range = self.index_range
        node.sparse_tensors = self.sparse_tensors
        self.generic_visit(node)
        return node

    # def visit_Subscript(self, node):
    #     '''
    #     For now just [:, None] and [None, :] are supported.
    #     '''
    #     self.generic_visit(node)
    #     assert ast.unparse(node.slice) in ['(:, None)', '(None, :)'], f"Unsupported slice: {ast.unparse(node.slice)}"
    #     node.indices = node.value.indices
    #     return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            node.indices = []
            for operand in [node.left, node.right]:
                assert hasattr(operand, 'indices'), ast.dump(operand) + ' in ' + ast.unparse(node)
                if len(operand.indices) > len(node.indices):
                    node.indices = operand.indices                
        elif isinstance(node.op, ast.MatMult):
            assert isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
            assert node.left.indices[1] == node.right.indices[0], f"Invalid indices for matmul: {node.left.indices} @ {node.right.indices}"
            node.indices = [node.left.indices[0], node.right.indices[1]]            
        elif isinstance(node.op, ast.Pow):
            assert isinstance(node.left, ast.Name)
            node.indices = node.left.indices
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
                full_indices = self.indices_map[node.args[0].id] + self.indices_map[node.args[1].id]
                has_repeating_index = False
                for index in full_indices:
                    if full_indices.count(index) > 1:
                        has_repeating_index = True
                        break
                assert has_repeating_index, "matmul operator does not contracting indices: " + ast.unparse(node)
                if has_repeating_index:
                    node.indices = full_indices
                # Remove the repeating indices
                node.indices = [x for x in full_indices if full_indices.count(x) == 1]
        return node

    def visit_Name(self, node):
        if node.id in self.indices_map:
            node.indices = self.indices_map[node.id]
        return node

    def visit_Constant(self, node):
        self.generic_visit(node)
        node.indices = []
        return node

    def visit_Assign(self, node):
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        self.generic_visit(node)
        if not hasattr(node.value, 'indices'):
            node.dont_transform = True
            return node
        if isinstance(node.value, ast.Call) and node.value.func.id in ['empty', 'empty_like']:
            print('dont transform', ast.unparse(node))
            node.dont_transform = True
            return node
        self.indices_map[target.id] = node.value.indices
        target.indices = node.value.indices
        #node.type_comment = 'indices: ' + str(indices)
        node.type_comment = f'target_indices: {target.indices}'
        return node

def transform(tree):
    return AttachIndexNotation().visit(tree)