import torch

student = torch.randn((3,4,5))
teacher = torch.randn((3,4,5))

print(student)
print(teacher)

sim = torch.cosine_similarity(student, teacher, dim = -1).mean()
print(sim)