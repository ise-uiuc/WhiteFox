
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.elu = torch.nn.ELU(alpha=1.0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.elu(v1)
        v3 = v2 / v1
        return v3
# Inputs to the model
x1 = torch.randn(2, 16, 10, 10)
