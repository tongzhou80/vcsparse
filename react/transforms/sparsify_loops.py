import ast

class ConvertDenseLoopToSparse(ast.NodeTransformer):
    def __init__(self, var, format):
        self.var = var
        self.format = format

    def visit_For(self, node):
        pass

class SparsifyLoops(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.index_range = node.index_range
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)

        if hasattr(node, 'orig_node'):
            orig_node = node.orig_node
            if hasattr(orig_node, 'sparse_info'):
                var, format = orig_node.sparse_info
                node = ConvertDenseLoopToSparse(var, format).visit(node)
        return node