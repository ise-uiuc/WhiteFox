
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        for i in range(1000):
            self.add_module(str(i), torch.nn.Conv2d(256, 256, 1, stride=1, padding=1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 2
        v3 = torch.clamp(v2, 0, 50)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.sigmoid(v5)

        for i in range(1000):
            v7 = getattr(self, str(i))(v6)
            v8 = v7 + 2
            v9 = torch.clamp(v8, 0, 50)
            v10 = v7 * v9
            v11 = v10 / 6
            v12 = self.sigmoid(v11)
            v6 = torch.cat((v6, v12), 1)

        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
