
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = torch.cat((v1, v2), dim=1)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 37, 43)
