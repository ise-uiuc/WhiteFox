
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = []
        v2 = []
        v3 = []
        v.append(torch.mm(x1, x1))
        v.append(x2 + torch.mm(x2, x3))
        if torch.sum(x1)!= 0:
            v2.append(x1)
        for j in range(10):
            if torch.sum(x2)!= 0:
                v3.append(x2)
        return torch.cat(v, 1) + torch.cat(v2, 1) + (v[0].clamp(max=20))
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
