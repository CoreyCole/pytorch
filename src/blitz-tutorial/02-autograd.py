"""
What is autograd?
- "Automatic Differentiation"
- The autograd package provides automatic differentiation for all operations on Tensors.
- It is a define-by-run framework
  - Your backprop is defined by how your code is run, and that every single iteration can be different
- Can track all computations on a Tensor
- To compute the derivatives, you can call .backward() on a Tensor that was tracked by `requires_grad`
- Generally speaking, torch.autograd is an engine for computing vector-Jacobian product

Here are math review notes on Gradients, Jacobian Matrices, and the Chain Rule:
http://mathonline.wikidot.com/gradients-jacobian-matrices-and-the-chain-rule-review
"""
import torch

"""Create a tensor and set requires_grad=True to track computation with it:"""
x = torch.ones(2, 2, requires_grad=True)

"""Do a tensor operation:"""
y = x + 2

"""y was created as a result of an operation, so it has a grad_fn"""
print(y.grad_fn)
# <AddBackward0 object at 0x7f341a4efef0>

"""example of vector-Jacobian product:"""
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
# tensor([-1109.3398,    -1.4602,   835.0535], grad_fn=<MulBackward0>)

"""Now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly,
but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:"""
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
# tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
