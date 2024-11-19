import ast
from ast_transforms.utils import *

class GenNumbaCode(ast.NodeTransformer):
    def __init__(self, parallel=False):
        self.parallel = parallel
        self.tensor_format = {}

    '''
    This transformer will need to handle arguments that Numba doesn't directly
    support, like sparse matrices, and convert them to a tuple of dense arrays.
    '''
    def visit_FunctionDef(self, node):
        '''
        Check the tensor arguments, and there are two situations:
        1. All dense tensors => simply add a @numba.njit decorator to the function
        2. At least one sparse tensor => convert it to a tuple of dense arrays, and 
        pass to a newly generated function, which is decorated with @numba.njit and 
        contains the actual function body
        '''
        sparse_tensors = {}
        for arg in node.args.args:
            varname = arg.arg
            indices = []
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                index_str = arg.annotation.args[0].value
                indices = index_str.split(',')
                if len(arg.annotation.args) > 1:
                    sparse_tensors[varname] = arg.annotation.args[1].value

        if len(sparse_tensors) == 0:
            node.decorator_list.append(new_ast_node_from_str(f"numba.njit(parallel={self.parallel})"))
        else:
            assert False, "Not implemented yet"
        return node

def transform(tree, *args):
    return GenNumbaCode(*args).visit(tree)