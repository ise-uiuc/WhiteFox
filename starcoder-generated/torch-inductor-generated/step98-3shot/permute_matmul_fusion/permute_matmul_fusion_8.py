
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = x1.permute(0, 2, 1)
        v4 = x1.permute(0, 2, 1)
        v5 = v1.permute(0, 2, 1)
        v6 = v2.permute(0, 2, 1)
        v7 = v3.permute(0, 2, 1)
        v8 = v4.permute(0, 2, 1)
        return torch.bmm(torch.bmm(v1, x1), torch.bmm(v2, x2))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
