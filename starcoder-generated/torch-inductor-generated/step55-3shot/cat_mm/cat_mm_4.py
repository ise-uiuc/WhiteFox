
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = []
        v.append(x1)
        v.append(x1)
        v.append(x1)
        v.append(x1)
        v.append(x1)
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
