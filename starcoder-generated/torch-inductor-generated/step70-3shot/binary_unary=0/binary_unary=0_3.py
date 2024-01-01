
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        a1 = self.conv2(v2)
        v3 = torch.relu(a1)
        a2 = self.conv3(v3)
        v4 = torch.relu(a2)
        a3 = self.conv1(v4)
        v5 = torch.relu(a3)
        a4 = self.conv2(v5)
        v6 = torch.relu(a4)
        a5 = self.conv3(v6)
        v7 = torch.relu(a5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
