
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.hardtanh(v1)
        v3 = torch.mul(v1, v2)
        v4 = self.conv2(v3)
        v5 = F.hardtanh(v4)
        v6 = torch.mul(v4, v5)
        v7 = v5 * v6
        v8 = self.conv3(v7)
        v9 = torch.mul(v8, v7)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
