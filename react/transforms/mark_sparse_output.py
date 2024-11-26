import ast

class MarkSparseOutput(ast.NodeTransformer):
    '''
    If the statement has form "B = A * ..." or "B = A / ...", and A is sparse, mark B as having sparse output
    '''
    def __init__(self):
        self.sparse_tensors = None

    def visit_FunctionDef(self, node):
        self.sparse_tensors = node.sparse_tensors
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Mult, ast.Div)):
            if node.value.left.id in self.sparse_tensors:
                #print("Marking", node.targets[0].id, "as sparse output")
                self.sparse_tensors[node.targets[0].id] = self.sparse_tensors[node.value.left.id]
        return node

def transform(node):
    return MarkSparseOutput().visit(node)
