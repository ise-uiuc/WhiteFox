
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(21, 19, (1, 3), stride=(1, 3), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(19, 17, (1, 3), stride=(1, 3), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(17, 15, (1, 3), stride=(1, 3), padding=(0, 1))
        self.conv4 = torch.nn.Conv2d(15, 13, (1, 3), stride=(1, 3), padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
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
        v13 = self.conv3(v12)
        v14 = self.conv4(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 38, 487, 614)
