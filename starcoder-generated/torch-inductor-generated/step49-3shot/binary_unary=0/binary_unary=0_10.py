
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1(x)
        v3 = x * v1
        v4 = torch.relu(v3)
        v5 = v4 + v2
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
