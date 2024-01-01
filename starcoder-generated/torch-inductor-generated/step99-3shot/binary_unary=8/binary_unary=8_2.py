
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 5, padding=2, stride=1, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = x1 * 3.7
        v3 = self.conv(v2) + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
