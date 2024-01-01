
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.functional.conv2d
        self.conv = self.conv2(128, 256, (17, 17), stride=2, padding=0)
        self.conv1 = self.conv2(256, 128, (17, 17), stride=1, padding=0)
        self.conv3 = self.conv2(-1, -1, (1, 1), stride=1, padding=0)
        self.conv4 = self.conv2(128, 10, (1, 1), stride=1, padding=0)
    def forward(self, x1):
        v2 = x1 * 0.5
        v21 = x1 * 0.7071067811865476
        v3 = torch.erf(v21)
        v4 = v3 + 1
        v5 = self.conv(v4)
        v6 = v5 * 0.5
        v7 = v5 * 0.7071067811865476
        v8 = torch.erf(v7)
        v9 = v8 + 1
        v10 = self.conv1(v9)
        v11 = v6 * v10
        v11 = torch.tanh(v11)
        v12 = self.conv3(v11)
        v13 = self.conv4(v12)
        return v13
# Inputs to the model
x1 = torch.randn(128, 1, 17, 17)
