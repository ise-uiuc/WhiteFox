
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = self.conv2(v4)
        v6 = v5 * 0.4472135955
        v7 = torch.erf(v6)
        v8 = v7 + 1
        v9 = self.conv3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 75, 75)
