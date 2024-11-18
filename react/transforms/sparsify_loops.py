import ast
from ast_transforms.utils import *

class ConvertDenseLoopToSparse(ast.NodeTransformer):
    def __init__(self, var, format):
        self.var = var
        self.format = format

    def visit_For(self, node):
        outer = node
        inner = None
        assert isinstance(node.body[0], ast.For)
        inner = node.body[0]
        outer_index = outer.target.id
        inner_index = inner.target.id
        inner.iter = new_ast_range(
            stop=new_ast_node_from_str(f'{self.var}.inptr[{outer_index}+1]'),
            start=new_ast_node_from_str(f'{self.var}.inptr[{outer_index}]'),
        )
        new_inner_index = '_' + inner_index
        inner.target = new_ast_name(new_inner_index)
        inner.body.insert(0, new_ast_assign(
            new_ast_name(inner_index, ctx=ast.Store()),
            new_ast_node_from_str(f'{self.var}.indices[{new_inner_index}]')
        ))
        RewriteSparseTensorRead(self.var, new_inner_index).visit(inner)
        return node

class RewriteSparseTensorRead(ast.NodeTransformer):
    '''
    Rewrite A[i,k] to A.data[_k] if A is a sparse tensor.
    '''
    def __init__(self, var, new_index):
        self.var = var
        self.new_index = new_index

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self.var:
            node = new_ast_node_from_str(f'{self.var}.data[{self.new_index}]')
        return node

class SparsifyLoops(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.index_range = node.index_range
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)

        if hasattr(node, 'orig_node'):
            orig_node = node.orig_node
            if hasattr(orig_node, 'sparse_info'):
                var, format = orig_node.sparse_info
                node = ConvertDenseLoopToSparse(var, format).visit(node)
        return node

def transform(node):
    return SparsifyLoops().visit(node)