import ast
from ast_transforms.utils import *

class DefinedVarVisitor(ast.NodeVisitor):
    def __init__(self):
        self.defined_vars = []

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id not in self.defined_vars:
                self.defined_vars.append(node.id)

class InsertAllocations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        assigns = []
        visitor = DefinedVarVisitor()
        visitor.visit(node)
        indices_map = node.indices_map
        index_range = node.index_range
        for v in visitor.defined_vars:
            shape = [f"{index_range[i][0]}.shape[{index_range[i][1]}]" for i in indices_map[v]]
            alloc = new_ast_assign(
                new_ast_name(v, ctx=ast.Store()),
                new_ast_call(
                    new_ast_name('empty'),
                    new_ast_node_from_str(f'({",".join(shape)})')
                )
            )
            alloc.targets[0].indices = indices_map[v]
            assigns.append(alloc)
        node.body = assigns + node.body
        return node

def transform(node):
    return InsertAllocations().visit(node)