
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 7, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(9, 12, 5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.cat([x1, v6], 1)
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 202, 74)
