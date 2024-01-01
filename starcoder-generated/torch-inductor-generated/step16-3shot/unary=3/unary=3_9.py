
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.maxpool(v2)
        v4 = self.conv(v3)
        v5 = self.relu(v4)
        v6 = self.maxpool(v5)
        v7 = self.conv(v6)
        v8 = self.relu(v7)
        v9 = self.maxpool(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
