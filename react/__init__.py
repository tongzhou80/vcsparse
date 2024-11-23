import sys
import ast
import inspect
import textwrap
import ast_transforms
from ast_transforms import apply_transform_on_ast
from .transforms import attach_index_notation, op_to_loop, trie_fuse, insert_allocations, parallelize
from .transforms import assign_sparse_to_dense, sparsify_loops, gen_numba_code, intraloop_scalar_replacement
from .transforms import remove_unused_array_stores, to_single_sparse_operand_form, to_inplace_sp_add_form
from .transforms import convert_matmul_op_to_call

def Index(*args):
    pass

def Tensor(*args):
    pass

def compile(**args):
    '''
    This decorator can either be annotated to a function without any arguments,
    or it can accept a list of arguments, in which case a new compile function is returned
    which compiles the function with the given arguments
    '''
    if len(args) == 1:
        return _compile(args[0])
    else:
        def _compile_fn(fn):
            return _compile(fn, **args)
        return _compile_fn

def _compile(fn, **options):
    newsrc = compile_from_src(inspect.getsource(fn), **options)
    header = textwrap.dedent('''
    import numba
    from numpy import empty, zeros
    ''')
    newsrc = header + newsrc
    if options.get("dump_code", False):
        print(newsrc)
    m = ast_transforms.utils.load_code(newsrc)
    return getattr(m, fn.__name__)

def compile_from_src(src, **options):
    if options.get("full_opt", False):
        options["gen_numba_code"] = True
        options["memory_opt"] = True
        options["trie_fuse"] = True
        options["parallelize"] = True
    tree = ast.parse(src)
    tree = apply_transform_on_ast(tree, "check_for_undefined", ["Tensor", "compile"])
    tree = apply_transform_on_ast(tree, "remove_func_decorator")
    tree = convert_matmul_op_to_call.transform(tree)
    tree = apply_transform_on_ast(tree, "to_single_op_form")
    if options.get("to_dense_first", False):
        tree = assign_sparse_to_dense.transform(tree)
    else:
        tree = to_single_sparse_operand_form.transform(tree)
    tree = attach_index_notation.transform(tree)
    tree = to_inplace_sp_add_form.transform(tree)
    tree = apply_transform_on_ast(tree, "attach_def_use_vars")
    tree = insert_allocations.transform(tree)
    tree = op_to_loop.transform(tree)
    tree = sparsify_loops.transform(tree)
    if options.get("trie_fuse", False):
        tree = trie_fuse.transform(tree)
    if options.get("gen_numba_code", False):
        if options.get("parallelize", False):
            tree = parallelize.transform(tree)
        tree = gen_numba_code.transform(tree, options.get("parallelize", False))
    if options.get("memory_opt", False):
        tree = intraloop_scalar_replacement.transform(tree)
        tree = remove_unused_array_stores.transform(tree)
    tree = apply_transform_on_ast(tree, "remove_func_arg_annotation")
    tree = apply_transform_on_ast(tree, "where_to_ternary")
    return ast_to_code(tree)

def ast_to_code(tree):
    return ast.unparse(tree).replace('# type:', '#')
