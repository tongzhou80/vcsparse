import ast
from ast_transforms.utils import *

class ArrayReferenceCheck(ast.NodeVisitor):
    def __init__(self, array_name, indices_str):
        self.array_name = array_name
        self.indices_str = indices_str
        self.loaded_times = 0
        self.always_same_index = True

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id == self.array_name:
            self.loaded_times += 1
            # Check if the array is always referenced with the same index
            if ast.unparse(node.slice) != self.indices_str:
                self.always_same_index = False
        return node


class IntraloopScalarReplacement(ast.NodeTransformer):
    def __init__(self):
        self.scalar_count = 0

    def visit_For(self, node):
        self.generic_visit(node)
        candidates = []
        for i, child in enumerate(node.body):
            # Check all top-level array store statements
            if isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Subscript):
                # Note that currently use_sparse_output only works in numba mode. In python mode, this place
                # will trigger an error
                #dump_code(child)
                target = child.targets[0]
                arrayname = target.value.id
                indices_str = ast.unparse(target.slice)
                #print(arrayname, indices_str)
                visitor = ArrayReferenceCheck(arrayname, indices_str)
                for successor in node.body[i+1:]:   # assume the definition reaches all statements after itself
                    visitor.visit(successor)

                if visitor.loaded_times > 0 and visitor.always_same_index:
                    candidates.append(target)
                    #print('candidate: ', ast.dump(target))

        # Scan the loop body and replace the subscripts with the generated scalar variables
        for sub in candidates:
            scalar_var = f'__scalar_{self.scalar_count}'
            self.scalar_count += 1
            ReplaceSubscriptsWithName(sub, scalar_var).visit(node)

            # Insert the stores at the end of the loop
            node.body.append(
                new_ast_assign(
                    deepcopy_ast_node(sub, ctx=ast.Store()),
                    new_ast_name(scalar_var)
                )
            )

        return node

class ReplaceSubscriptsWithName(ast.NodeTransformer):
    def __init__(self, subscript, name):
        self.subscript = subscript
        self.name = name

    def visit_Subscript(self, node):
        if ast.unparse(node) == ast.unparse(self.subscript):
            if isinstance(node.value.ctx, ast.Load):
                return new_ast_name(self.name)
            else:
                return new_ast_name(self.name, ctx=ast.Store())
        return node

def transform(node):
    return IntraloopScalarReplacement().visit(node)