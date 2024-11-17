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
        self.top_level_loops = []
        self.trie_fuse = trie_fuse
        self.index_range = None
        self.indices_map = None

    def visit_FunctionDef(self, node):
        self.index_range = node.index_range
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def get_loop_by_index(self, index, loops):
        for loop in loops:
            if loop.target.id == index:
                return loop
        return None

    def get_index_bound(self, i):
        rg = self.index_range[i]
        return f'{rg[0]}.shape[{rg[1]}]'

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and node.value.func.id in ['empty', 'zeros', 'ones']:
            return node
        target = node.targets[0]
        assert isinstance(target, ast.Name)

        indices = []
        for v in node.def_vars + node.use_vars:            
            for i in self.indices_map[v]:
                if i not in indices:
                    indices.append(i)

        orig_node = deepcopy_ast_node(node)
        new_stmt = NameToSubscript(self.indices_map).visit(node)
        new_stmt = RemoveNoneAxis().visit(new_stmt)
        loop = new_ast_perfect_for(
            [new_ast_name(i) for i in indices],
            [new_ast_range(new_ast_node_from_str(self.get_index_bound(i))) for i in indices],
            [new_stmt]
        )
        loop.orig_node = orig_node

        # is_reduction = False
        # if isinstance(node.value, ast.Call) and node.value.func.id in ['sum', 'max', 'min', 'matmul']:
        #     is_reduction = True

        if isinstance(node.value, ast.Call) and node.value.func.id in ['sum', 'max', 'min', 'matmul']:
            if node.value.func.id == 'matmul':
                reduction_index = self.indices_map[node.use_vars[0]][1]
            else:
                axis = node.value.args[1].value                
                reduction_index = self.indices_map[node.use_vars[0]][axis]
            # Mark that loop level as reduction
            loop = MarkLoopAsReduction(reduction_index).visit(loop)
            # Insert initialization at proper place
            initialization = new_ast_assign(
                node.targets[0],
                get_init_value_for_reduction(node.value.func.id)
            )
            initialization_indices = self.indices_map[node.def_vars[0]]
            initialization_loop = new_ast_perfect_for(
                [new_ast_name(i) for i in initialization_indices],
                [new_ast_range(new_ast_node_from_str(self.get_index_bound(i))) for i in initialization_indices],
                [initialization]
            )
            #loop = InsertInitialization(self.indices_map[node.def_vars[0]], initialization).visit(loop)
            # Fix reduction assignment
            loop = FixReductionAssign().visit(loop)
            return initialization_loop, loop
        else:
            return loop

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

class FixReductionAssign(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and node.value.func.id in ['sum', 'max', 'min', 'matmul']:
            if node.value.func.id  == 'sum':
                node.value = new_ast_add(
                    deepcopy_ast_node(node.targets[0], ctx=ast.Load()),
                    node.value.args[0],
                )
            elif node.value.func.id == 'matmul':
                node.value = new_ast_add(
                    deepcopy_ast_node(node.targets[0], ctx=ast.Load()),
                    new_ast_mul(
                        node.value.args[0],
                        node.value.args[1],
                    )
                )
            elif node.value.func.id in ['max', 'min']:
                node.value = new_ast_call(
                                new_ast_name(node.value.func.id),
                                [deepcopy_ast_node(node.targets[0], ctx=ast.Load()), 
                                 node.value.args[0]],
                            )
            else:
                assert False
        return node

class MarkLoopAsReduction(ast.NodeTransformer):
    def __init__(self, reduction_index):
        self.reduction_index = reduction_index

    def visit_For(self, node):
        if node.target.id == self.reduction_index:
            node.is_reduction = True
        self.generic_visit(node)
        return node

def transform(node, trie_fuse=False):
    return OpToLoop(trie_fuse).visit(node)