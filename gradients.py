import torch

print('-'*50)

x = torch.tensor([2.0,3.0,4.0], requires_grad=True)
print('x', x)

y = x*2+1
print(y)
z = y.mean()
print(z)

z.backward()
print('Gradient of x = ',x.grad)