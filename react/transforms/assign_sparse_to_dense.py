import ast
from ast_transforms.utils import *

class AssignSparseToDense(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        '''
        Record the storage format of the arguments based on annotations
        '''
        self.tensor_format = {}
        for arg in node.args.args:
            varname = arg.arg
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                # The default format is dense
                self.tensor_format[varname] = 'dense'
                if len(arg.annotation.args) > 1:
                    self.tensor_format[varname] = arg.annotation.args[1].value
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        '''
        Check if the assignment uses any operand that is sparse, and seperate it out 
        to a new assignment if so.
        '''
        visitor = ReplaceSparseOperands(self.tensor_format)
        visitor.visit(node)
        return visitor.new_stmts + [node]
        
class ReplaceSparseOperands(ast.NodeTransformer):
    def __init__(self, tensor_format):
        self.tensor_format = tensor_format
        self.new_stmts = []
        self.var_count = 0

    def get_new_var(self):
        self.var_count += 1
        return '__dt%d' % self.var_count  # stands for "dense temporary"

    def visit_Name(self, node):
        if node.id in self.tensor_format:
            if self.tensor_format[node.id] in ('csr', 'csc'):
                # We need to make a dense temporary
                # and assign the sparse operand to it
                # and then replace the original operand with the temporary
                node.sparse_format = self.tensor_format[node.id]
                tmp = self.get_new_var()
                new_assign = new_ast_assign(
                    new_ast_name(tmp, ctx=ast.Store()),
                    node
                )
                new_assign.sparse_info = (node.id, self.tensor_format[node.id])
                self.new_stmts.append(
                    new_assign
                )
                return new_ast_name(tmp)

        return node

def transform(node):
    return AssignSparseToDense().visit(node)