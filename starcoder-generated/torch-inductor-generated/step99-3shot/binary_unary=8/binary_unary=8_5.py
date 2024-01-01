
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 192, 131, padding=0, stride=11, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
x2 = torch.randn(1, 16, 16, 16)
