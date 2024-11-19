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

        node.decorator_list.append(new_ast_node_from_str(f"numba.njit(parallel={self.parallel})"))
        if len(sparse_tensors) > 0:
            # Create an outer function that launches the inner function (node)
            outer_func = new_ast_function_def(
                name=node.name,
                args=deepcopy_ast_node(node.args),
                body=[]
            )
            node.name = f"_{node.name}"
            orig_args = deepcopy_ast_node(node.args)
            # Update the arguments of the inner function (node) such that every sparse tensor is replaced
            # by 3 dense arrays, e.g. indptr, indices and data
            newargs = []
            for arg in orig_args.args:
                varname = arg.arg
                if varname in sparse_tensors:
                    newargs.append(new_ast_arg(f"{varname}_indptr"))
                    newargs.append(new_ast_arg(f"{varname}_indices"))
                    newargs.append(new_ast_arg(f"{varname}_data"))
                else:
                    newargs.append(arg)
            node.args.args = newargs
            # Add a call statement in outer_func to call the inner function with the dense arrays
            actual_args = []
            for arg in orig_args.args:
                varname = arg.arg
                if varname in sparse_tensors:
                    actual_args.append(new_ast_name(f"{varname}.indptr"))
                    actual_args.append(new_ast_name(f"{varname}.indices"))
                    actual_args.append(new_ast_name(f"{varname}.data"))
                else:
                    actual_args.append(arg)
            
            outer_func.body.append(
                new_ast_return(
                    new_ast_call(
                        new_ast_name(node.name),
                        actual_args
                    )
                )
            )

            #dump_code(node)
            #dump_code(outer_func)
            #assert False, "Unimplemented"
        return node, outer_func

def transform(tree, *args):
    return GenNumbaCode(*args).visit(tree)