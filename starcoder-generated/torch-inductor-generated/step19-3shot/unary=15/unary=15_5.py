
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.avg = torch.nn.AdaptiveAvgPool2d(8)
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avg(v1)
        v3 = self.flatten(v2)
        v4 = torch.softmax(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
