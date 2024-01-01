
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = v2.permute(0, 2, 1)
        v4 = v1.permute(0, 2, 1)
        v5 = torch.bmm(v0, v4)
        v6 = torch.matmul(v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
