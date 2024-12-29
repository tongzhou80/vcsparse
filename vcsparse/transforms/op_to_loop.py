import ast
from ast_transforms.utils import *

class NameToSubscript(ast.NodeTransformer):
    def __init__(self, indices_map):
        self.indices_map = indices_map

    def visit_Name(self, node):
        indices = self.indices_map.get(node.id, [])
        if indices:
            subscript = new_ast_subscript(
                value=node,
                indices=[new_ast_name(i) for i in indices],
                ctx=node.ctx
            )
            return subscript
        else:
            return node


class OpToLoop(ast.NodeTransformer):
    def __init__(self, trie_fuse=False):
        self.top_level_loops = []
        self.trie_fuse = trie_fuse
        self.indices_map = None

    def visit_FunctionDef(self, node):
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def get_loop_by_index(self, index, loops):
        for loop in loops:
            if loop.target.id == index:
                return loop
        return None

    def get_bound(self, t):
        assert len(t) == 3 and t[0] in ('dense', 'sparse')
        return f'{t[1]}.shape[{t[2]}]'

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and node.value.func.id in ['empty', 'empty_like', 'zeros', 'ones', 'csr_matrix']:
            return node
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        # Skip assignments that don't participate in the transformation
        if target.id not in self.indices_map:
            return node

        indices = []
        index_range = {}
        for v in node.use_vars + node.def_vars: 
            for pos,idx in enumerate(self.indices_map[v]):
                if idx not in indices:
                    indices.append(idx)
                    # Make the iteration space dense by default
                    index_range[idx] = ('dense', v, pos)
        #index_range = node.iter_space_info
        #print(index_range)     

        # Sanity check
        for i in target.indices:
            assert i in indices, f"Target index {i} not in indices: {indices}" + str(self.indices_map)

        orig_node = deepcopy_ast_node(node)
        new_stmt = NameToSubscript(self.indices_map).visit(node)
        loop = new_ast_perfect_for(
            [new_ast_name(i) for i in indices],
            [new_ast_range(new_ast_node_from_str(f'{self.get_bound(index_range[i])}')) for i in indices],
            [new_stmt]
        )
        loop.orig_node = orig_node

        FixTransposedAccesses().visit(loop)

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
                deepcopy_ast_node(node.targets[0]),
                get_init_value_for_reduction(node.value.func.id)
            )
            initialization_indices = self.indices_map[node.def_vars[0]]
            initialization_loop = new_ast_perfect_for(
                [new_ast_name(i) for i in initialization_indices],
                [new_ast_range(new_ast_node_from_str(self.get_bound(index_range[i]))) for i in initialization_indices],
                [initialization]
            )
            #loop = InsertInitialization(self.indices_map[node.def_vars[0]], initialization).visit(loop)
            # Fix reduction assignment
            loop = FixReductionAssign().visit(loop)
            return initialization_loop, loop
        else:
            return loop

class FixTransposedAccesses(ast.NodeTransformer):
    '''
    This rewrites A[i,j] to A[j,i] if A has attribute "is_transpose"
    '''
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and hasattr(node.value, 'is_transpose'):
            # Swap the indices in the slice
            elts = node.slice.elts
            assert len(elts) == 2
            node.slice.elts = [elts[1], elts[0]]
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