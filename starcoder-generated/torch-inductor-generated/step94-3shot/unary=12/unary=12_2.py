
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 1, stride=1, padding=1, dilation=2)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.gelu(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 4, 32)
