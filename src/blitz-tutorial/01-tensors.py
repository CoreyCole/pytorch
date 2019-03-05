"""
What is PyTorch?
- A replacement for NumPy to use the power of GPUs
- A deep learning research platform that provides maximum flexibility and speed

Tensors
- Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
"""
import torch
import numpy as np

"""Construct a 5x3 matrix, uninitialized:"""
x = torch.empty(5, 3)
# tensor([[-1.1015e-07,  4.5807e-41, -3.5803e-06],
#         [ 4.5807e-41,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])

"""Construct a randomly initialized matrix:"""
x = torch.rand(5, 3)
# tensor([[0.9727, 0.8805, 0.5857],
#         [0.1284, 0.4006, 0.1293],
#         [0.5041, 0.9665, 0.4347],
#         [0.0748, 0.1356, 0.5269],
#         [0.2129, 0.0561, 0.2891]])

"""Construct a matrix filled zeros and of dtype long:"""
x = torch.zeros(5, 3, dtype=torch.long)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

"""Construct a tensor directly from data:"""
x = torch.tensor([5.5, 3])
# tensor([5.5000, 3.0000])

"""Create a tensor based on an existing tensor:
(carries over properties like dtype)"""
x = x.new_ones(5, 3, dtype=torch.double)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)

"""Override dtype for new tensor:"""
x = torch.randn_like(x, dtype=torch.float)
# tensor([[ 0.3323, -0.8246,  1.5832],
#         [ 0.1779, -0.0341, -0.3968],
#         [-0.4179,  1.3846,  1.0981],
#         [-1.5426, -0.4654, -0.6574],
#         [-0.9417, -0.0177, -1.1593]])

"""Resizing: If you want to resize/reshape tensor, you can use torch.view:"""
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
# torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

"""Interface with numpy ndarrays:
(The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.)"""
# from torch to numpy
a = torch.ones(5)
b = a.numpy()
# from numpy to torch
c = np.ones(5)
d = torch.from_numpy(c)

"""CUDA tensors"""
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
