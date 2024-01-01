
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = F.dropout(x1, p=0.5)
        x4 = torch.rand_like(x2)
        x5 = F.dropout(x4)
        x6 = torch.softmax(x3, dim=1)
        x7 = F.dropout(x6)
        x8 = torch.log_softmax(x7, dim=1)
        return x8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 3, 2)
