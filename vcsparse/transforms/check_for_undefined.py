import ast

'''
This is a simple undefined variable check without using reaching definition.
It simply creates a set of variables that are defined in the function body + 
function arguments. It then checks all loaded Names, and see if they are in the set.
'''

class GetDefinedVariables(ast.NodeVisitor):
    def __init__(self):
        self.defined = set()

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.defined.add(arg.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        return node

class CheckForUndefined(ast.NodeVisitor):
    def __init__(self, defined_set):
        self.defined_set = defined_set

    def visit_FunctionDef(self, node):
        for child in node.body:
            self.visit(child)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.defined_set:
                raise NameError(f'name `{node.id}` is undefined: line {node.lineno} at offset {node.col_offset}')
        return node

def transform(node):
    visitor = GetDefinedVariables()
    visitor.visit(node)
    builtin_names = [
        'Tensor', 
        'empty', 'zeros', 'ones',
        'matmul', 'sum', 'max', 'min',
        'pow', 'log', 'exp', 'sin', 'tan',
        'where'
    ]
    visitor = CheckForUndefined(list(visitor.defined) + builtin_names)
    visitor.visit(node)
    return node