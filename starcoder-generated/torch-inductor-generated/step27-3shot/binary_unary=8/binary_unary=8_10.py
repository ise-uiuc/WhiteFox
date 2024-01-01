
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 4, 1, stride=2)
        self.conv3 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv2(x1)
        v4 = v1 + v2
        v5 = v3 + v2
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
