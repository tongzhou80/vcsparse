import sys
import ast
import inspect
import textwrap
import ast_transforms
from ast_transforms import apply_transform_on_ast
from .transforms import attach_index_notation, op_to_loop, trie_fuse, insert_allocations, parallelize
from .transforms import assign_sparse_to_dense, sparsify_loops, gen_numba_code, intraloop_scalar_replacement
from .transforms import remove_unused_array_stores, to_single_sparse_operand_form, to_inplace_sp_add_form
from .transforms import check_for_undefined, convert_matmul_op_to_call, remove_none_axis, mark_transpose_ops
from .transforms import convert_sparse_multiply_call, fix_sparse_operand_to_left, attach_iter_space_info
from .transforms import mark_sparse_output

def Index(*args):
    pass

def Tensor(*args):
    pass

def compile(fn=None, **args):
    '''
    This decorator can either be annotated to a function without any arguments,
    or it can accept a list of arguments, in which case a new compile function is returned
    which compiles the function with the given arguments
    '''
    # If no compiler options are given, just return the compiled function
    if fn != None:
        return _compile(fn)
    # Otherwise return a function that accepts a function as an argument, and also pass the compiler arguments
    else:
        def _compile_fn(f):
            return _compile(f, **args)
        return _compile_fn

def _compile(fn, **options):
    newsrc = compile_from_src(inspect.getsource(fn), **options)
    header = textwrap.dedent('''
    import numba
    from numpy import empty, zeros, matmul, empty_like
    from scipy.sparse import csr_matrix
    ''')
    newsrc = header + newsrc
    if options.get("dump_code", False):
        print(newsrc)
    m = ast_transforms.utils.load_code(newsrc)
    return getattr(m, fn.__name__)

def compile_from_src(src, **options):
    if options.get("full_opt", False):
        if "gen_numba_code" not in options:
            options["gen_numba_code"] = True
        if "memory_opt" not in options:
            options["memory_opt"] = True
        if "trie_fuse" not in options:
            options["trie_fuse"] = True
        if "parallelize" not in options:
            options["parallelize"] = True
    
    tree = ast.parse(src)
    tree = check_for_undefined.transform(tree)
    tree = apply_transform_on_ast(tree, "remove_func_decorator")
    tree = convert_matmul_op_to_call.transform(tree)
    tree = convert_sparse_multiply_call.transform(tree)
    tree = mark_transpose_ops.transform(tree)
    tree = apply_transform_on_ast(tree, "to_single_op_form")
    if options.get("preprocess_only", False):
        tree = apply_transform_on_ast(tree, "remove_func_arg_annotation")
        return ast_to_code(tree)

    if options.get("to_dense_first", False):
        tree = assign_sparse_to_dense.transform(tree)
    else:
        tree = to_single_sparse_operand_form.transform(tree)
    tree = remove_none_axis.transform(tree)
    tree = fix_sparse_operand_to_left.transform(tree)
    tree = attach_index_notation.transform(tree)
    if options.get("use_sparse_output", False):
        tree = mark_sparse_output.transform(tree)
    tree = to_inplace_sp_add_form.transform(tree)
    tree = apply_transform_on_ast(tree, "attach_def_use_vars")
    tree = attach_iter_space_info.transform(tree)
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

    tree = apply_transform_on_ast(tree, "where_to_ternary")
    tree = apply_transform_on_ast(tree, "remove_func_arg_annotation")
    return ast_to_code(tree)

def ast_to_code(tree):
    return ast.unparse(tree).replace('# type:', '#')
