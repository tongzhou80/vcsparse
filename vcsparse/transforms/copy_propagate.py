import ast
from ast_transforms.utils import *

class ReplaceName(ast.NodeTransformer):
    def __init__(self, replace_map):
        self.replace_map = replace_map

    def visit_Name(self, node):
        if node.id in self.replace_map:
            node.id = self.replace_map[node.id]
        return node


class CopyPropagate(ast.NodeTransformer):
    def __init__(self):
        self.replacements = {}

    def visit_Assign(self, node):
        visitor = ReplaceName(self.replacements)
        visitor.visit(node)

        # If the statement has form `a = __b`, then replace all occurences of `a` with `__b` in subsequent statements
        if isinstance(node.value, ast.Name) and node.value.id.startswith('__'):
            self.replacements[node.targets[0].id] = node.value.id
            return None  # Remove this node itself
        else:
            return node

    def visit_Return(self, node):
        visitor = ReplaceName(self.replacements)
        visitor.visit(node)
        return node

    def visit_Expr(self, node):
        visitor = ReplaceName(self.replacements)
        visitor.visit(node)
        return node

    def visit_For(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_Try(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_AsyncFor(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_With(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_AsyncWith(self, node):
        # Clear the replacement map
        self.replacements.clear()
        self.generic_visit(node)
        return node

    def visit_Break(self, node):
        # Clear the replacement map
        self.replacements.clear()
        return node

    def visit_Continue(self, node):
        # Clear the replacement map
        self.replacements.clear()
        return node


def transform(tree):
    return CopyPropagate().visit(tree)