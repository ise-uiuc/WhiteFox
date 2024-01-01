
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise = torch.nn.Conv2d(1, 1, 1, stride=1)
        self.depthwise = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, groups=1)
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1)
    def forward(self, x):
        v1 = self.pointwise(x)
        v2 = self.depthwise(v1)
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 142, 142)
