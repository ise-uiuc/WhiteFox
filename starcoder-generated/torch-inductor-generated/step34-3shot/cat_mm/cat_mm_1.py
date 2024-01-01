
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t = []
        t.append(torch.cat([torch.mm(x1, x2) for _ in range(2)], 1))
        t.append(torch.cat([torch.mm(x1, x2) for _ in range(3)], 1))
        t.append(torch.cat([torch.mm(x1, x2) for _ in range(2)], 1))
        t.append(torch.cat([torch.mm(x1, x2) for _ in range(5)], 1))
        return torch.cat(t, 1)
# Inputs to the model
x1 = torch.randn(5, 2)
x2 = torch.randn(2, 3)
