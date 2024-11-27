import ast
from ast_transforms.utils import *

class MarkSparseOutput(ast.NodeTransformer):
    '''
    If the statement has form "B = A * ..." or "B = A / ...", and A is sparse, mark B as having sparse output
    '''
    def __init__(self):
        self.sparse_tensors = None
        self.func_body = None
        self.sparse_like = {}
        self.new_stmts = []

    def visit_FunctionDef(self, node):
        self.sparse_tensors = node.sparse_tensors
        self.func_body = node.body
        self.generic_visit(node)

        for s in self.new_stmts:
            self.func_body.insert(0, s)

        FixSparseReturns(self.sparse_like).visit(node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Mult, ast.Div)) and isinstance(node.value.left, ast.Name):
            if node.value.left.id in self.sparse_tensors and ast.unparse(node.value.right) not in self.sparse_tensors:
                #print("Marking", node.targets[0].id, "as sparse output")
                node.use_sparse_output = True
                left = node.value.left.id
                target = node.targets[0].id
                self.sparse_tensors[target] = self.sparse_tensors[left]
                self.sparse_like[target] = left
                sp_inits = [
                    new_ast_node_from_str(f'{target}_data = empty_like({left}.data)', inline=False),
                    new_ast_node_from_str(f'{target}_shape = {left}.shape', inline=False),
                    new_ast_node_from_str(f'{target}_indices = {left}.indices', inline=False),
                    new_ast_node_from_str(f'{target}_indptr = {left}.indptr', inline=False),
                ]
                for s in sp_inits:
                    s.dont_transform = True
                self.new_stmts.extend(sp_inits)

        return node

class FixSparseReturns(ast.NodeTransformer):
    def __init__(self, sparse_like):
        self.sparse_like = sparse_like

    def visit_Return(self, node):
        if node.value.id in self.sparse_like:
            other = self.sparse_like[node.value.id]
            node.value = new_ast_node_from_str(f'({node.value.id}_data, {other}_indices, {other}_indptr)')
        return node

def transform(node):
    return MarkSparseOutput().visit(node)
