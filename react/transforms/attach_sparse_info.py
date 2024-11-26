import ast

class AttachSparseInfo(ast.NodeTransformer):
    def __init__(self):
        self.sparse_tensors = None

    def visit_FunctionDef(self, node):
        self.sparse_tensors = node.sparse_tensors
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        sparse_vars = {}
        for var in node.use_vars + node.def_vars:
            if var in self.sparse_tensors:
                sparse_vars[var] = self.sparse_tensors[var]

        index_range = {}
        for var, format in sparse_vars.items():
            indices = self.indices_map[var]
            if format == 'csr':
                # For CSR format, the second index is indexed by the first index
                if indices[1] in index_range:
                    assert index_range[indices[1]] == (var, indices[0]), "conflicting sparse ranges for index " + str(indices[1])
                index_range[indices[1]] = (var, indices[0])
            elif format == 'csc':
                if indices[0] in index_range:
                    assert index_range[indices[0]] == (var, indices[1]), "conflicting sparse ranges for index " + str(indices[0])
                # For CSC format, the first index is indexed by the second index
                index_range[indices[0]] = (var, indices[1])
            else:
                assert False, "Unsupported sparse format: " + format

        node.sparse_tensors = sparse_vars
        node.sparse_index_ranges = index_range

def transform(node):
    return AttachSparseInfo().visit(node)