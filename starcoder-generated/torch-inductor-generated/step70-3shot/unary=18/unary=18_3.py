
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.depthconv = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1, groups=1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.depthconv(v3)
        v5 = torch.sigmoid(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
