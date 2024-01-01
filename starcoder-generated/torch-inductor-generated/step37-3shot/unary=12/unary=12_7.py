
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.maxpool(self.relu(self.bn(self.conv(x1))))
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
