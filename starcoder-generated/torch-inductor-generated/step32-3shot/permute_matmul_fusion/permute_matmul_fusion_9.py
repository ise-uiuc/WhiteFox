
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.matmul(v0, v1)
        v3 = torch.bmm(v1[..., 0], v0[..., 1])
        v4 = torch.cat((v2, v3), 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
