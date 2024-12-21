import ast
from ast_transforms.utils import *

class TrieFuse(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.fuse_loops_in_body(node)

        for child in node.body:
            if isinstance(child, ast.While):
                self.fuse_loops_in_body(child)
        return node

    def do_fusion_group(self, node, loops):
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
            # Try to fuse with the subsequent loops
            for loop in loops[(i+1):]:
                if host.target.id == loop.target.id:
                    host.body.extend(loop.body)
                    loops_to_be_removed.append(loop)
                    fused_loops_count += 1
                else:
                    break
            
            # Move to the next host which is `fused_loops_count` loops ahead
            i += fused_loops_count
            i += 1

        for loop in loops_to_be_removed:
            node.body.remove(loop)
        
        # Repeat the fusion process for the new loops
        for child in node.body:
            if isinstance(child, ast.For):
                self.fuse_loops_in_body(child)


    def fuse_loops_in_body(self, node):
        # Find continuous loops in the body, each group becomes a fusion group
        loops = []
        in_fusion_group = False
        for child in node.body:
            if isinstance(child, ast.For):
                loops.append(child)
                in_fusion_group = True
            else:
                if in_fusion_group:
                    # End a fusion group when a non-loop statement is encountered
                    self.do_fusion_group(node, loops)
                    loops = []
                    in_fusion_group = False

        self.do_fusion_group(node, loops)

        
def transform(node):
    return TrieFuse().visit(node)
