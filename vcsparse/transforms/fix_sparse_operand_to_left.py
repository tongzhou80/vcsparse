import ast

class MakeLeftSparse(ast.NodeTransformer):
    def __init__(self):
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
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            # If the left operand is dense and the right operand is sparse, swap them
            if isinstance(node.value.right, ast.Name) and node.value.right.id in self.sparse_tensors:
                node.value.left, node.value.right = node.value.right, node.value.left
        return node

def transform(node):
    return MakeLeftSparse().visit(node)