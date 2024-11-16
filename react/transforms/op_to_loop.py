import ast
from ast_transforms.utils import *

class NameToSubscript(ast.NodeTransformer):
    def __init__(self, indices_map):
        self.indices_map = indices_map

    def visit_Name(self, node):
        newnode = node
        if node.id in self.indices_map:
            if self.indices_map[node.id]:
                subscript = new_ast_subscript(
                    value=node,
                    indices=[new_ast_name(i) for i in self.indices_map[node.id]],
                )
                newnode = subscript
        return newnode

class OpToLoop(ast.NodeTransformer):
    def __init__(self, trie_fuse=False):
        self.loops = []
        self.trie_fuse = trie_fuse
        self.index_range = None
        self.indices_map = None

    def visit_FunctionDef(self, node):
        self.index_range = node.index_range
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        indices = node.targets[0].indices
        for i in indices:
            loop = new_ast_for(
                new_ast_name(i),
                new_ast_range(new_ast_node_from_str(self.index_range[i])),
            )
            #dump_code(loop)
        loop = new_ast_perfect_for(
            [new_ast_name(i) for i in indices],
            [new_ast_range(new_ast_node_from_str(self.index_range[i])) for i in indices],
            [NameToSubscript(self.indices_map).visit(node)]
        )
        
        #dump_code(new_assign)
        dump_code(loop)
        if not self.trie_fuse:
            pass
        return node

def transform(node, trie_fuse=False):
    return OpToLoop(trie_fuse).visit(node)