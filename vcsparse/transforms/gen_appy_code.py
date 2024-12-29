from ast_transforms.utils import *
import ast_comments as ast

class AnnotateOuterLoop(ast.NodeTransformer):
    def __init__(self, *args):
        pass

    def visit_FunctionDef(self, node):
        # Starting with _ means a kernel function
        if node.name.startswith('_'):
            node.decorator_list.append(new_ast_node_from_str(f"appy.jit"))
            self.generic_visit(node)    
        return node

    def visit_For(self, node):
        if not hasattr(node, 'is_reduction'):
            comment = ast.Comment(value='#pragma parallel for', inline=False)
            return comment, node
        return node


class LoopFinder(ast.NodeVisitor):
    def __init__(self):
        self.loops = []

    def visit_For(self, node):
        self.generic_visit(node)
        self.loops.append(node)


class AnnotateInnerLoop(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        # If the loop is an innermost loop, annotate it with `#pragma simd`
        if not hasattr(node, 'is_reduction'):
            visitor = LoopFinder()
            visitor.visit(node)

            # FIXME: The sparse loop cannot be vectorized for now due to a conflicting type error for the index if reassigned
            if len(visitor.loops) == 1: # and not node.target.id.startswith('__p'):
                comment = ast.Comment(value='#pragma simd(128)', inline=False)
                return comment, node

        return node

def transform(tree, *args):
    tree = AnnotateOuterLoop(*args).visit(tree)
    tree = AnnotateInnerLoop(*args).visit(tree)
    return tree