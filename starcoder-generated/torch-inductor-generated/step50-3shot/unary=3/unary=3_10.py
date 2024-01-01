
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(53, 7, 7, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(7, 53, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv5(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 53, 13, 12)
