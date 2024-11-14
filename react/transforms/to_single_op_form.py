import ast

class BinaryOpToAssign(ast.NodeTransformer):
    def __init__(self):
        self.stmts = []
        self.var_count = 0

    def get_new_var(self):
        self.var_count += 1
        return '__v%d' % self.var_count

    def visit_BinOp(self, node):
        newleft = self.visit(node.left)
        newright = self.visit(node.right)
        # newleft and newright now may be statements
        if isinstance(newleft, ast.Assign):
            node.left = ast.Name(id = newleft.targets[0].id, ctx = ast.Load())
        else:
            node.left = newleft

        if isinstance(newright, ast.Assign):
            node.right = ast.Name(id = newright.targets[0].id, ctx = ast.Load())
        else:
            node.right = newright

        assign = ast.Assign(targets = [ast.Name(id = self.get_new_var(), ctx = ast.Store())], value = node, lineno = node.lineno, col_offset = node.col_offset)
        self.stmts.append(assign)
        return assign


class ToSingleOperatorStmts(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.BinOp):
            visitor = BinaryOpToAssign()
            assign = visitor.visit(node.value)
            node.value = assign.targets[0]
            return visitor.stmts + [node]
        else:
            return node

class ReturnExprToStmt(ast.NodeTransformer):
    def visit_Return(self, node):
        if not isinstance(node.value, ast.Name):
            assign = ast.Assign(targets = [ast.Name(id = '__ret', ctx = ast.Store())], value = node.value, lineno = node.lineno, col_offset = node.col_offset)
            node.value = ast.Name(id = '__ret', ctx = ast.Load())
            return [assign] + [node]
        else:
            return node


class RemoveRedundantAssign(ast.NodeTransformer):
    def __init__(self):
        self.prev = None

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Name) and node.value.id.startswith('__v'):
            assert self.prev != None and self.prev.targets[0].id == node.value.id
            self.prev.targets[0] = node.targets[0]
            return
        else:
            self.prev = node
            return node


def transform(tree):
    tree = ReturnExprToStmt().visit(tree)
    tree = ToSingleOperatorStmts().visit(tree)
    tree = RemoveRedundantAssign().visit(tree)
    return tree