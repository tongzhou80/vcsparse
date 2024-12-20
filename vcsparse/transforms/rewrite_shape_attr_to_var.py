import ast
from ast_transforms.utils import *

class AttributeToName(ast.NodeTransformer):
    def __init__(self):
        self.attr_to_name = {}

    def visit_Subscript(self, node):
        '''
        This function rewrites attributes such as `A.shape[0]` to `A_shape_0`
        '''
        self.generic_visit(node)
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'shape' and isinstance(node.ctx, ast.Load):
            name = f'{node.value.value.id}_shape_{ast.unparse(node.slice)}'
            self.attr_to_name[ast.unparse(node)] = name
            return ast.Name(id=name, ctx=ast.Load())

        return node


class RewriteShapeAttrToVar(ast.NodeTransformer):
    '''
    This pass rewrites attributes such as `A.shape[0]` to `A_shape_0` when used in for ... range statements.
    New assign statements will be inserted which define the shape variables.
    '''
    def __init__(self):
        self.new_stmts = []
        self.defined_vars = set()

    def visit_FunctionDef(self, node):        
        self.generic_visit(node)
        # Insert the new statements before the first loop in the body
        pos = 0
        for stmt in node.body:
            if isinstance(stmt, ast.For) or isinstance(stmt, ast.While):
                break
            pos += 1
        node.body = node.body[:pos] + self.new_stmts + node.body[pos:]
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                new_stmts = []
                visitor = AttributeToName()
                node.iter = visitor.visit(node.iter)
                for attr, name in visitor.attr_to_name.items():
                    if name not in self.defined_vars:
                        self.defined_vars.add(name)
                        self.new_stmts.append(
                            new_ast_assign_from_str(f'{name} = {attr}')
                        )
        return node


def transform(node):
    return RewriteShapeAttrToVar().visit(node)
    