import ast
from ast_transforms import apply_transform_on_ast
from .transforms import attach_index_notation, op_to_loop

def Index(*args):
    pass

def compile_from_src(src, **options):
    tree = ast.parse(src)
    tree = apply_transform_on_ast(tree, "to_single_op_form")
    tree = attach_index_notation.transform(tree)
    tree = apply_transform_on_ast(tree, "add_func_decorator", "numba.jit")
    tree = apply_transform_on_ast(tree, "remove_func_arg_annotation")
    return ast_to_code(tree)

def ast_to_code(tree):
    return ast.unparse(tree).replace('# type:', '#')
