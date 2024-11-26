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
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        assigns = []
        visitor = DefinedVarVisitor()
        visitor.visit(node)
        indices_map = node.indices_map
        index_range = node.index_range
        for v in visitor.defined_vars:
            # If `v` is a function argument, skip it
            if v in args:
                continue
            # If `v` does not participate in the transformation, skip it
            if v not in indices_map:
                continue
            shape = [f"{index_range[i][0]}.shape[{index_range[i][1]}]" for i in indices_map[v]]
            alloc_func = 'empty'
            alloc = new_ast_assign(
                new_ast_name(v, ctx=ast.Store()),
                new_ast_call(
                    new_ast_name(alloc_func),
                    new_ast_node_from_str(f'({",".join(shape)})')
                )
            )
            alloc.targets[0].indices = indices_map[v]
            assigns.append(alloc)
        node.body = assigns + node.body
        return node

def transform(node):
    return InsertAllocations().visit(node)