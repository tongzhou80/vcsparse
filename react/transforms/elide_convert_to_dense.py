'''
It's not always necessary to convert the sparse tensors to a dense temporary. 
The only situation when this is necessary is when there is a binary operation
and both operands are sparse.

- Unary operations where only A is sparse
    - sum(A)
- Binary operations where only A is sparse
    - A + 1 
    - A + B

Sparse-Dense additions need to be converted to a form that uses in-place update to save memory.
Like C = A + 1 is converted to C = 1; C += A. Then when codegen for "C += A", it's safe
to just use a sparse iteration space.

What about sparse to dense assignment? which is equivalent to C = A + "default_value".
It can also be converted to the in-place update form: C = "default_value"; C += A.

Can we say that once the additions are converted to in-place updates form, we can elide 
always converting the sparse tensors to dense temporaries?

C = spA + spB => well, still each statement can have at most one sparse operand.
C = spA + spB => dA = spA; C = dA + spB
              => dA = spA; C = dA; C += spB
              => dA = 0; dA += spA; C = dA; C += spB
              => now every statement has a non-conflicting iteration space

C = A + spB   => dA = A; C = dA; C += spB
              => dA = 0; dA += A; C = dA; C += spB

C = spA * B   => this one needs no conversion. Just use the spA's iteration space.
              => wait, C is dense, so its other values need to be zeroed out.
              => C = 0; C += spA * B
              => looks like a spA * B is just like spA alone in terms of iteration space

C = spA * spB => dA = spA; C = dA * spB
              => dA = 0; dA += spA; C = 0; C += dA * spB

C = spA @ B   => reduction pattern will be recognized, convert to C = 0; C += spA @ B
              => then "C += spA @ B" also has non-conflicting iteration spaces

C = sum(spA, 1)  => reduction pattern will be recognized, convert to C = 0; C += sum(spA, 1)

C = spA @ spB => dB = spB; C = spA @ dB
              => dB = 0; dB += spB; C = 0; C += spA @ dB
              
So the formula is converting
    C = (dense or const) + sparse * (dense or const) 
=>  C = (dense or const); C += sparse * (dense or const);
=>  (single dense iter space); (single sparse iter space)

What about unary operations? For now we just assume that unary element-wise operations are
only gonna apply to the nonzero elements. And same for unary reduction operations. 
So under this assumption, it's safe to always use a single sparse iteration space for unary operations.

So the conclusion is:
Without dense format conversion elision, we always convert every sparse tensor to its dense form first, and
use the dense form for all operations (iteration spaces will be always dense later on).
With this elision, we convert operations according to the formula above. For the resulting operations,
they should either have dense iteration space, or a single sparse iteration space, so no conflicting anyways.

An extra optimization is that for now temporary tensors are always dense, but in some situations, they can
use sparse formats, if such formats are known, like for C = spA * B.
'''