
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 16, 5, stride=3, padding=4)
        self.conv2 = torch.nn.Conv2d(16, 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(4, 21, (1, 1), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(21, 8, (5, 5), stride=(2, 2), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = self.conv(v9)
        v11 = v10 * 0.5
        v12 = v10 * 0.7071067811865476
        v13 = torch.erf(v12)
        v14 = v13 + 1
        v15 = v11 * v14
        v16 = self.conv2(v15)
        v17 = self.conv3(v16)
        return v17
# Inputs to the model
x1 = torch.randn(2, 17, 21, 33)
