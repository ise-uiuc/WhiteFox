
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(21, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(21, 64, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v1 = torch.relu(v1)
        v2 = torch.matmul(x2, v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.matmul(x3, v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(21, 21)
x3 = torch.randn(21, 21)
