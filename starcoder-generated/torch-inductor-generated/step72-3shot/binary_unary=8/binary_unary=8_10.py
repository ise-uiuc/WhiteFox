
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.conv3(v1)
        v6 = self.conv4(v2)
        v7 = self.conv3(v1)
        v8 = self.conv4(v2)
        v9 = v3 + v4 + v5 + v6 + v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
