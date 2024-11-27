import ast
from ast_transforms.utils import *


class GenNumbaCode(ast.NodeTransformer):
    def __init__(self, parallel=False):
        if parallel:
            self.parallel = True
        else:
            self.parallel = False
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
        sparse_tensors = node.sparse_tensors
        indices_map = node.indices_map

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
                    newargs.append(new_ast_arg(f"{varname}_shape"))
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
                    actual_args.append(new_ast_name(f"{varname}.shape"))
                else:
                    actual_args.append(arg)
            
            outer_return = new_ast_return(
                    new_ast_call(
                        new_ast_name(node.name),
                        actual_args
                    )
                )
            if '__ret' in sparse_tensors:
                old_value = outer_return.value
                outer_return.value = new_ast_call(
                    new_ast_name('csr_matrix'),
                    [old_value]
                )

            outer_func.body.append(outer_return)


            node = RewriteAttributeWithName(sparse_tensors).visit(node)

            #dump_code(node)
            #dump_code(outer_func)
            #assert False, "Unimplemented"
            return node, outer_func
        else:
            return node

class RewriteAttributeWithName(ast.NodeTransformer):
    def __init__(self, sparse_tensors):
        self.sparse_tensors = sparse_tensors

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.sparse_tensors:
            return new_ast_name(f"{node.value.id}_{node.attr}")
        return node
        #if node.attr in self.sparse_tensors:
            

def transform(tree, *args):
    return GenNumbaCode(*args).visit(tree)