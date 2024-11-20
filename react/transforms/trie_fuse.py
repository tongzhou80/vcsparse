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

        loops_to_be_removed = []
        i = 0
        while i < len(loops):
            host = loops[i]
            if hasattr(host, 'is_reduction'):
                i += 1
                continue
            
            fused_loops_count = 0
            for loop in loops[(i+1):]:
                if host.target.id == loop.target.id:
                    host.body.extend(loop.body)
                    loops_to_be_removed.append(loop)
                    fused_loops_count += 1
                else:
                    break

            i += fused_loops_count
            i += 1

        for loop in loops_to_be_removed:
            node.body.remove(loop)
        
        # Repeat the fusion process for the new loops
        for child in node.body:
            if isinstance(child, ast.For):
                self.fuse_loops_in_body(child)

def transform(node):
    return TrieFuse().visit(node)
