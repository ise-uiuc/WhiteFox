
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, [1, 3, 3, 7], stride=[1, 1, 4, 1], padding=[1, 0, 0, 0])
        self.conv2 = torch.nn.Conv2d(16, 16, [1, 5, 1, 21], stride=[1, 1, 4, 1], padding=[1, 2, 0, 0])
        self.conv3 = torch.nn.Conv2d(16, 16, [1, 7, 7, 5], stride=[1, 1, 4, 1], padding=[1, 0, 0, 0])
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x3
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = x3 + v2
        v7 = torch.relu(v6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 86, 64)
x3 = torch.randn(1, 16, 64, 64)
