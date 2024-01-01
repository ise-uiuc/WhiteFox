
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 20, (1, 1), stride=(1, 1), padding=(0, 0))
        self.t1 = torch.nn.Transpose2d((1, 2), (1, 0))
        self.conv3 = torch.nn.Conv2d(4, 30, (5, 5), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(30, 14, (1, 1), stride=(2, 2), padding=(0, 0))
        self.t2 = torch.nn.Transpose2d((2, 3), (1, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.t1(v1)
        v3 = self.conv3(v2)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv4(v8)
        v10 = v9 + 0.5
        v11 = v9 + 1
        v12 = torch.sqrt(v3)
        v13 = v11 * v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 32, 17, 17)
