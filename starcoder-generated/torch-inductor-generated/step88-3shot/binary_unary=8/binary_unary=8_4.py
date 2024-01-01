
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
