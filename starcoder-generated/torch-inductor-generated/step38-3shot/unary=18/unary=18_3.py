
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.relu = torch.nn.ReLU(True)
        self.pool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn = torch.nn.BatchNorm2d(num_features=16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.pool2x2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 448, 448)
