
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 15, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(15, 7, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(7, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        return self.conv3(v12)
# Inputs to the model
x1 = torch.randn(1, 12, 28, 28)
