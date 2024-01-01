
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = torch.matmul(x1.permute(0, 2, 1), x2)
        v1 = torch.bmm(x2, v0)
        v2 = v0.permute(0, 2, 1)
        v3 = torch.bmm(v2, v1)
        v4 = torch.cat((v0, v3), 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
