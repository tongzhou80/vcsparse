import ast
from .transforms import to_single_op_form, attach_index_notation

def Index(*args):
    pass

def compile_from_src(src, **options):
    tree = ast.parse(src)
    tree = to_single_op_form.transform(tree)
    tree = attach_index_notation.transform(tree)
    return ast_to_code(tree)

def ast_to_code(tree):
    return ast.unparse(tree).replace('# type:', '# ')
