import torch

# x = torch.ones(2, 2, requires_grad=True)
# print(x)

# y = x + 2
# print(y)

# z = 3 * y**2
# out = z.mean()
# print(z, out)

# # requires_grad flag defaults to False
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

# # Let's backprop now. Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.))
# out.backward()
# print(x.grad)  # d(out)/dx_i Jacobian matrix

# torch.autograd is an engine for computing vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

'''Now in this case y is no longer a scalar.
 torch.autograd could not compute the full Jacobian directly,
 but if we just want the vector-Jacobian product, simply
 pass the vector to backward as argument:
 '''
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# you can disable automatic history tracking on tensors using the code block:
with torch.no_grad():
    print((x ** 2).requires_grad)