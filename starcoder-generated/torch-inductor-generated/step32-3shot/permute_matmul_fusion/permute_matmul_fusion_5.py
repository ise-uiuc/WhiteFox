
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = torch.bmm(X1, v0)
        v2 = torch.bmm(v0, X1)
        v3 = (v1.permute(0, 1, 2) * v2.permute(1, 2, 0)).permute(0, 2, 1)
        v4 = torch.cat((v3, X2), 1)[..., 0]
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
