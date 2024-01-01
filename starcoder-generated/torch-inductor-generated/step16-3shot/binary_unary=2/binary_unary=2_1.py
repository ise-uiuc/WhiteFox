
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.depthwise = torch.nn.Conv2d(32, 32, 3, stride=1, groups=32, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.depthwise(v1)
        v3 = v2 - 0.2
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
