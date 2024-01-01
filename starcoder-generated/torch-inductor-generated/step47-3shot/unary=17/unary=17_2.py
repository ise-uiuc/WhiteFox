
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.conv(v3)
        v5 = self.maxpool(v4)
        v6 = self.conv(v5)
        v7 = self.relu(v6)
        v8 = self.conv(v7)
        v9 = self.relu(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
