
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.randn(2, 2)
        v2 = torch.randn(2, 2)
        v3 = torch.randn(2, 2)
        v4 = torch.randn(2, 2)
        v5 = torch.randn(2, 2)
        v6 = torch.randn(2, 2)
        v7 = torch.bmm(v1, v3)
        v8 = torch.bmm(v2, v4)
        v9 = torch.bmm(v5, v7)
        v10 = torch.bmm(v6, v8)
        return torch.bmm(v9, v10)
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)
