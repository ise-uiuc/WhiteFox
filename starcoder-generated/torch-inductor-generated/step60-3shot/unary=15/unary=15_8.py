
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.add(v1, v2)
        v4 = torch.mul(v2, v3)
        v5 = torch.max(v4, v3)
        v6 = self.conv3(v5)
        v7 = self.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(4, 32, 224, 224)
