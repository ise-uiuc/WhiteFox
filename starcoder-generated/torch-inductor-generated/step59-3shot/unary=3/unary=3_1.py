
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 7, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(7, 14, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(14, 10, 5, stride=2, padding=0)
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
        v11 = self.conv3(v9)
        return v11
# Inputs to the model
x1 = torch.randn(1, 8, 57, 57)
