
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 9, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(9, 8, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.tanh(v6)
        v8 = self.conv2(v7)
        v9 = self.conv3(v8)
        v10 = self.conv4(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 72, 84)
