
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2) for i in range(3)], 1)
        v2 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v3 = torch.cat([torch.mm(x1, x2) for i in range(3)], 1)
        return torch.cat([v1, v2, v3], 1)
# Inputs to the model
x1 = torch.randn(6, 3)
x2 = torch.randn(3, 1)
