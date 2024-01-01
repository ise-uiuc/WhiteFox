
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        return v7 + v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
