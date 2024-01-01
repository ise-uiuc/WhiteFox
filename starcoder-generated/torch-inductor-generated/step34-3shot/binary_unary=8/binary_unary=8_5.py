
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        for _ in range(50):
            v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv1(v1)
        v4 = self.conv1(v1)
        v5 = v2 + v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
