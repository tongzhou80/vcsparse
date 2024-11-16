import ast
from ast_transforms.utils import *

class TrieFuse(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.fuse_loops_in_body(node)
        return node

    def fuse_loops_in_body(self, node):
        loops = []
        for child in node.body:
            if isinstance(child, ast.For):
                loops.append(child)

        if len(loops) == 0:
            return
        
        host = None
        # Make the first non-reduction loop host
        for i,l in enumerate(loops):
            if not hasattr(l, 'is_reduction'):
                host = l
                break

        if host is None:
            return

        loops_to_be_removed = []
        for loop in loops[(i+1):]:  # starting from the (i+1)th loop
            if host.target.id == loop.target.id:
                host.body.extend(loop.body)
                loops_to_be_removed.append(loop)

        for loop in loops_to_be_removed:
            node.body.remove(loop)
        
        self.fuse_loops_in_body(host)

def transform(node):
    return TrieFuse().visit(node)
