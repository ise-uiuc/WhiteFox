
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        a1 = self.conv2(v2)
        v3 = torch.relu(a1)
        v4 = a1 + v3
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = v6 + v3
        v8 = self.conv3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
