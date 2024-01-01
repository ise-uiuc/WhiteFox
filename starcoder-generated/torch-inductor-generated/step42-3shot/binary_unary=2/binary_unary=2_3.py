
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=0)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.1
        v3 = F.relu(v2)
        v4 = self.bn(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 128)
