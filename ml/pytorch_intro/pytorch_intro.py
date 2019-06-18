from __future__ import print_function
import torch
import numpy as np

# # zero matrix (float type)
# x = torch.empty(5,3)
# print(x)

# # zero matrix (long(integer) type)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# random matrix (entries are floats between 0 and 1)
# x = torch.rand(5,3)
# print(x)

# # creat tensor with specified entries
# x = torch.tensor([5.5, 3])
# print(x)

# # reuse propertieso of the input type (old x) such as data type
# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)

# x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x)                                      # result has the same size

# print(x.size())

y = torch.rand(5, 3)
# print(x + y)			# method 1: addition
# print(torch.add(x, y))	# method 2: addition

# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

# result = torch.empty(5,3)
# result.add_(x)				# add in-place

# -------------------------------------------------------------------------------------------------------------------------------
# **NOTE** Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
# -------------------------------------------------------------------------------------------------------------------------------


# # Resizing: If you want to resize/reshape tensor, you can use torch.view:
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x)
# print(y)
# print(z)

# # Converting between numpy array and pytorch tensor
# a = np.ones(5)
# b = torch.tensor(a)
# c = b.numpy()
# print(a)
# print(b)
# print(c)

# # changing the tensor also changes c, the numpy array obtained from that tensor 
# b.add_(4)
# print(a)
# print(b)
# print(c)




