
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(1, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(3, 12, 1, stride=1, padding=0, dilation=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = self.conv(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
