from cypath.pytorch.matrix import *
import torch

cm = torch.tensor([[6258,1780],[7540,21345]])
print(metrics(cm))
cm = torch.tensor([[11581,4407],[28373,103932]])
print(metrics(cm))
cm = torch.tensor([[16792,7234],[31248,129942]])
print(metrics(cm))
