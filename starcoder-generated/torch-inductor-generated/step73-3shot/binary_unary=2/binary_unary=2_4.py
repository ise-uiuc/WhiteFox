
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 64, 1, stride=1, groups=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 4.0
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
