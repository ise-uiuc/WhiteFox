
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        x = F.dropout(input)
        y = x[0]
        if y:
            x.append(5)
        else:
            x.pop()
        del x[0]
        return F.dropout(input)
# Inputs to the model
input = torch.randn(1)
