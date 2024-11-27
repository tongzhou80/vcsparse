import ast

class ConvertMatmulOpToCall(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            return ast.Call(func=ast.Name(id='matmul', ctx=ast.Load()), args=[node.left, node.right], keywords=[])
        return self.generic_visit(node)

def transform(node):
    return ConvertMatmulOpToCall().visit(node)