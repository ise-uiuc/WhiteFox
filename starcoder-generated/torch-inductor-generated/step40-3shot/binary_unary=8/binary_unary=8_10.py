
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1, groups=3)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1, groups=3)
        self.conv4 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v1)
        v4 = self.conv2(v1)
        v5 = self.conv3(x1)
        v6 = self.conv3(x1)
        v7 = self.conv4(v5)
        v8 = self.conv4(v5)
        v9 = v3 + v4 + v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
