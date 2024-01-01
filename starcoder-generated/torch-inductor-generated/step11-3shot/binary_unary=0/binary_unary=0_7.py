
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = torch.sigmoid(x1)
        v2 = self.conv1(v1)
        v3 = torch.relu(x2)
        v4 = v2 + v3
        v5 = self.conv2(v4)
        v6 = v5 + x3
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = 0
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
