import ast
from ast_transforms.utils import *

'''
This pass renames loops that share the same index name to different names.

Input:

for i in range(...):
    ...
for i in range(...):
    ...

Output:

for i0 in range(...):
    ...
for i1 in range(...):
    ...

This pass only updates dense indices, so if the loop is sparse like this:
for __pB_i in range(...):
    j = B_indices[__pB_i]
    ...

It'll get updated to
for __pB_i in range(...):
    j0 = B_indices[__pB_i]
    ...

'''
class ReplaceName(ast.NodeTransformer):
    def __init__(self, replace_map):
        self.replace_map = replace_map

    def visit_Name(self, node):
        if node.id in self.replace_map:
            node.id = self.replace_map[node.id]
        return node


class RenameLoopIndices(ast.NodeTransformer):
    def __init__(self):
        self.index_count = {}

    def visit_For(self, node):
        self.generic_visit(node)
        if not node.target.id.startswith('__'):
            dense_index = node.target.id
        else:
            dense_index = node.body[0].targets[0].id
            
        count = self.index_count.get(dense_index, 0)
        self.index_count[dense_index] = count + 1
        new_dense_index = dense_index + str(count)
        ReplaceName({dense_index: new_dense_index}).visit(node)
        return node

def transform(tree):
    return RenameLoopIndices().visit(tree)