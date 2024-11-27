import ast
from ast_transforms.utils import *

class MarkSparseOutput(ast.NodeTransformer):
    '''
    If the statement has form "B = A * ..." or "B = A / ...", and A is sparse, mark B as having sparse output
    '''
    def __init__(self):
        self.sparse_tensors = None
        self.func_body = None
        self.new_stmts = []

    def visit_FunctionDef(self, node):
        self.sparse_tensors = node.sparse_tensors
        self.func_body = node.body
        self.generic_visit(node)

        for s in self.new_stmts:
            self.func_body.insert(0, s)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Mult, ast.Div)):
            if node.value.left.id in self.sparse_tensors and node.value.right.id not in self.sparse_tensors:
                #print("Marking", node.targets[0].id, "as sparse output")
                node.use_sparse_output = True
                left = node.value.left.id
                target = node.targets[0].id
                self.sparse_tensors[target] = self.sparse_tensors[left]
                sp_init = new_ast_node_from_str(f'{target} = csr_matrix((empty_like({left}.data), {left}.indices, {left}.indptr), shape={left}.shape)', inline=False)
                self.new_stmts.append(sp_init)
        return node

def transform(node):
    return MarkSparseOutput().visit(node)
