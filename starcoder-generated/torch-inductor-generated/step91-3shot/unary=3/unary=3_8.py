
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 7, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 2, 2, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(2, 3, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v6)
        v9 = self.conv4(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 28, 12)
