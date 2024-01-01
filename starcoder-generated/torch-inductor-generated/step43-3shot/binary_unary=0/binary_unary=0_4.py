
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = self.conv2(v1) + y
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = v4 + v1
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)
