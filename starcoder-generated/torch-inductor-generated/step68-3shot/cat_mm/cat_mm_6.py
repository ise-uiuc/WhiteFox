
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        v8 = torch.mm(x1, x2)
        v9 = torch.mm(x1, x2)
        v10 = torch.mm(x1, x2)
        return torch.cat([v1, v2, v3, v8, v9, v10, v1, v2, v3], 1)
# Inputs to the model
x1 = torch.randn(3, 1)
x2 = torch.randn(1, 2)
