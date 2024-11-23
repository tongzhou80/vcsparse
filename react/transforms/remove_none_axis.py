import ast

class RemoveNoneAxis(ast.NodeTransformer):
    '''
    This class removes code patterns such as 'a[:, None]' or 'a[None, :]'
    '''
    def visit_Subscript(self, node):
        node_str = ast.unparse(node)
        if node_str.endswith('[:, None]') or node_str.endswith('[None, :]'):
            return node.value
        else:
            return node

def transform(tree):
    return RemoveNoneAxis().visit(tree)