
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 2, stride=2, padding=1)
        self.conv_bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_bn(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 1024)
