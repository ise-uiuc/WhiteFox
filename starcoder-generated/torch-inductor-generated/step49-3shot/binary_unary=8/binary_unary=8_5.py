
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = self.conv2(v1)
        v6 = self.conv2(v2)
        v7 = self.conv2(v3)
        v8 = self.conv2(v4)
        v9 = v5 + v6 + v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
