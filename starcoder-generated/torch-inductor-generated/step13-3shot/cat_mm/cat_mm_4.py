
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        v4 = torch.mm(x1, x2)
        v5 = torch.mm(x1, x2)
        v1_1 = torch.mm(v1, v2)
        v3_1 = torch.mm(v3, v4)
        v5_1 = torch.mm(v5, v1)
        v3_2 = torch.mm(v1, v4)
        return torch.cat([v1_1, v3_1, v5_1, v1, v2, v3, v4, v5], 1)
# Inputs to the model
x1 = torch.randn(1, 9)
x2 = torch.randn(9, 1)
