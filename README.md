VCSparse: Generate fused and vectorizable kernels for sparse tensor programs on CPUs and GPUs. An web tool is available to try VCSparse online: [https://tongzhou80.github.io/vcsparse-web/index.html](https://tongzhou80.github.io/vcsparse-web/index.html).

# Installation

```bash
pip install vcsparse
```

# Quick introduction

To compile a function with `vcsparse`, all you need to do is to add *index notation* to the function arguments, 
and decorate the function with `@vcsparse.compile`. Here's an example:

```python
@vcsparse.compile(full_opt=True)
def mul_then_rowwise_sum(A: Tensor('i,j', 'csr'), B: Tensor('i,j', 'csr')):
    return (A * B).sum(1)
```

The option `full_opt=True` will enable all optimization passes. 