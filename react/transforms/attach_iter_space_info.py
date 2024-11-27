import ast
from ast_transforms.utils import *
from collections import OrderedDict

class AttachSparseInfo(ast.NodeTransformer):
    def __init__(self):
        self.sparse_tensors = None
        self.indices_map = None

    def visit_FunctionDef(self, node):
        self.sparse_tensors = node.sparse_tensors
        self.indices_map = node.indices_map
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if hasattr(node, 'dont_transform'):
            return node

        indices = []
        iteration_spaces = {}
        for v in node.use_vars + node.def_vars: 
            for pos,idx in enumerate(self.indices_map[v]):
                if idx not in indices:
                    indices.append(idx)
                    # Make the iteration space dense by default
                    iteration_spaces[idx] = ('dense', v, pos)

        sparse_vars = OrderedDict()
        for var in node.def_vars + node.use_vars:
            if var in self.sparse_tensors:
                sparse_vars[var] = self.sparse_tensors[var]

        # This might assign iter space to an index multiple times, so an index will get its
        # iter space from the last assignment
        for var, format in sparse_vars.items():
            indices = self.indices_map[var]
            if format == 'csr':
                # For CSR format, the second index is indexed by the first index
                #if indices[1] in iteration_spaces:
                #    assert iteration_spaces[indices[1]] == (var, indices[0]), "conflicting sparse ranges for index " + str(indices[1])
                iteration_spaces[indices[1]] = ('sparse', var, indices[0])
            elif format == 'csc':
                #if indices[0] in iteration_spaces:
                #    assert iteration_spaces[indices[0]] == (var, indices[1]), "conflicting sparse ranges for index " + str(indices[0])
                # For CSC format, the first index is indexed by the second index
                iteration_spaces[indices[0]] = ('sparse', var, indices[1])
            else:
                assert False, "Unsupported sparse format: " + format

        node.sparse_tensors = sparse_vars
        node.iter_space_info = iteration_spaces
        node.indices = indices
        # dump_code(node)
        # print(node.sparse_tensors)
        # print(node.sparse_iteration_spacess)
        self.generic_visit(node)
        return node

def transform(node):
    return AttachSparseInfo().visit(node)