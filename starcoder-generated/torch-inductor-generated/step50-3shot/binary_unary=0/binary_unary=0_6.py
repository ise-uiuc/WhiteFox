
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1 + x2)
        v2 = torch.relu(v1)
        v3 = self.conv3(x3 + x4)
        v4 = torch.relu(v3)
        v5 = self.conv2(v2 + v4)
        v6 = v5 + v1
        v7 = torch.relu(v6)
        return torch.nn.functional.max_pool2d(v7, kernal_size=3, stride=1, padding=1)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
