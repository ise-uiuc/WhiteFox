
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v1, v2)
        v4 = torch.bmm(v1, v2)
        v5 = v3 + v4
        v6 = torch.add(v1, v5)
        v7 = v6.permute(0, 2, 1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
