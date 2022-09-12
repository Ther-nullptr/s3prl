import torch

a = torch.tensor([1,2,3]).float()
print(torch.softmax(a, dim = 0))