
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.swish = torch.nn.SiLU(inplace=True)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = torch.ops.aten.silu(v1)
        v4 = 3 + v2
        v5 = torch.clamp(v4, 0., 6.)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7.unsqueeze(0)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
