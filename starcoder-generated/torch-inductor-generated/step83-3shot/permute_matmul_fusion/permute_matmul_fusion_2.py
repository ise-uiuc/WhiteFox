
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        v1 = torch.bmm(x1.permute(0, 2, 1), v0)
        v2 = v1.permute(0, 2, 1)[0][0][1]
        v3 = x1 * x2
        v4 = x1 * v1
        return v2 + v3 + v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
