
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 96, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v5)
        v8 = v7 + v6
        return torch.relu(v8)
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
