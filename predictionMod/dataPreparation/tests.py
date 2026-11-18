import torch

a = torch.tensor((1, 2))
a[0] = 1
a[1] = 2

print(a)
a = torch.stack((a,a))
print(a)