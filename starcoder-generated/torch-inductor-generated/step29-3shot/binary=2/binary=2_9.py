
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, groups=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, groups=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0, groups=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0, groups=1)
    def forward(self, x):
        v1 = self.relu(x)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = (v2 + v5) - v1
        return v6
# Inputs to the model
x = torch.randn(1, 16, 32, 32)
