
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = x1.permute(0, 2, 1)
        v4 = torch.bmm(v1, v3)
        v5 = torch.bmm(v4, v0)
        v6 = torch.bmm(x1, x2)
        v7 = torch.cat((v7, v6), 2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
