
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 + y
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = v6 + v2
        return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)
