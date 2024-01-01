
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 5, 4, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 14, 4, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(14, 5, 4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 + 0.3090169943749475
        v9 = torch.cos(v8)
        v10 = torch.flatten(v9, 1)
        v11 = self.conv3(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 8, 25, 25)
