
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = torch.matmul(x1.permute(0, 2, 1), x2.permute(0, 2, 1))
        v1 = v0.permute(0, 2, 1)
        v2 = torch.bmm(v1, v1)
        v3 = torch.bmm(v2, v2)
        v4 = torch.bmm(v3, v3)
        v5 = torch.bmm(v4, v4)
        return (v0, v1, v2, v3, v4, v5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
