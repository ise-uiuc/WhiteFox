
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = torch.matmul(v0, x1)
        v2 = torch.matmul(v0, v1)
        v3 = v2.permute(0, 2, 1)
        v4 = torch.matmul(v0, v1)
        v5 = torch.matmul(x1, v3)
        v6 = torch.cat((v5, v4), 1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
