
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = []
        i1 = torch.mm(x1, x2) + torch.mm(x1, x3)
        i2 = torch.mm(x1, x2) + torch.mm(x1, x3)
        i3 = torch.mm(x1, x2) + torch.mm(x1, x3)
        v.append(i1)
        v.append(i2)
        v.append(i3)
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(2, 5)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 3)
