
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv2(v2)
        v4 = self.conv1(v3)
        v5 = self.conv2(v4)
        v6 = self.conv1(v5)
        v7 = v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
