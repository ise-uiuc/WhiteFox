
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=10, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=20, padding=0)
        self.conv6 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv6(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
