
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=8)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=4)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = v1 + x1
        v5 = self.conv1(v4)
        v6 = self.conv2(v5)
        v7 = v6 + v5 + v3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
