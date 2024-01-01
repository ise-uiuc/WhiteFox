
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d()
        self.conv = torch.nn.Conv2d(3, 128, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 10, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(10)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = self.conv(x1)
        v3 = self.conv2(v2)
        v4 = 3 + v3
        v5 = torch.clamp(v4, 0, 6)
        v6 = v3 * v5
        v7 = v6 / 6
        v8 = self.bn(v7)
        v9 = self.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
