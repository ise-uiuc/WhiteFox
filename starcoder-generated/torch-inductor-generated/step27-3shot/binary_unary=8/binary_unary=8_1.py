
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv5(x1)
        v4 = self.conv5(x1)
        v5 = torch.cat([v1, v2, v3, v4], axis=1)
        v6 = self.conv6(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
