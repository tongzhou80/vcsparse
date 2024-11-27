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
        # Make the original inner loop have sparse iteration space
        inner.iter = new_ast_range(
            stop=new_ast_node_from_str(f'{self.var}.indptr[{outer_index}+1]'),
            start=new_ast_node_from_str(f'{self.var}.indptr[{outer_index}]'),
        )
        new_inner_index = f'__p{self.var}_{outer_index}'
        inner.target = new_ast_name(new_inner_index)
        inner.body.insert(0, new_ast_assign(
            new_ast_name(inner_index, ctx=ast.Store()),
            new_ast_node_from_str(f'{self.var}.indices[{new_inner_index}]')
        ))
        RewriteSparseTensorRead(self.var, new_inner_index).visit(inner)
        
        return node

class ConvertDenseLoopToSparseNew(ast.NodeTransformer):
    def __init__(self, iter_space_info, sparse_tensors):
        self.iter_space_info = iter_space_info
        self.sparse_tensors = sparse_tensors

    def visit_For(self, node):
        self.generic_visit(node)
        is_info = self.iter_space_info
        dense_or_sparse, tensor, index = is_info[node.target.id]
        if dense_or_sparse == 'sparse':
            node.iter = new_ast_range(
                stop=new_ast_node_from_str(f'{tensor}.indptr[{index}+1]'),
                start=new_ast_node_from_str(f'{tensor}.indptr[{index}]'),
            )
            old_index = node.target.id
            new_index = f'__p{tensor}_{index}'
            node.target = new_ast_name(new_index)
            node.body.insert(0, new_ast_assign(
                new_ast_name(old_index, ctx=ast.Store()),
                new_ast_node_from_str(f'{tensor}.indices[{new_index}]')
            ))
            for st in self.sparse_tensors:
                RewriteSparseTensorRead(st, new_index).visit(node)

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
            if hasattr(orig_node, 'iter_space_info'):
                node = ConvertDenseLoopToSparseNew(
                            orig_node.iter_space_info,
                            orig_node.sparse_tensors
                            ).visit(node)
        return node

def transform(node):
    return SparsifyLoops().visit(node)