import ast

class CheckArrayReferences(ast.NodeVisitor):
    def __init__(self):
        self.loaded_arrays = []
        self.stored_arrays = []

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            varname = node.value.id
            if isinstance(node.ctx, ast.Load) and varname not in self.loaded_arrays:
                self.loaded_arrays.append(varname)
            elif isinstance(node.ctx, ast.Store) and varname not in self.stored_arrays:
                self.stored_arrays.append(varname)

        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.loaded_arrays:
            self.loaded_arrays.append(node.id)
        elif isinstance(node.ctx, ast.Store) and node.id not in self.stored_arrays:
            self.stored_arrays.append(node.id)

        return node

class RemoveStoreToArray(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Assign(self, node):
        # Check if the target is an array store and if it's in self.names
        # If so, remove the node
        target = node.targets[0]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.names:
            return None

        return node

class RemoveUnusedArrayStores(ast.NodeTransformer):
    '''
    If an array is stored but never loaded (as in subscripts), and
    it's not a function argument, remove the store.
    '''
    def visit_FunctionDef(self, node):
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        visitor = CheckArrayReferences()
        visitor.visit(node)
        arrays_to_remove = []
        for varname in visitor.stored_arrays:
            if varname not in args and varname not in visitor.loaded_arrays:
                arrays_to_remove.append(varname)

        transformer = RemoveStoreToArray(arrays_to_remove)
        node = transformer.visit(node)
        return node

def transform(node):
    transformer = RemoveUnusedArrayStores()
    node = transformer.visit(node)
    return node