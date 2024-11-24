import ast

class MarkTransposeOps(ast.NodeTransformer):
    '''
    1. Convert A.T to A, but the returned node will be added a new attribute is_transpose = True
    2. Convert transpose(A) to A, but the returned node will be added a new attribute is_transpose = True
    '''
    def visit_Attribute(self, node):
        if node.attr == 'T':
            node.value.is_transpose = True
            return node.value
        else:
            return node

def transform(node):
    return MarkTransposeOps().visit(node)