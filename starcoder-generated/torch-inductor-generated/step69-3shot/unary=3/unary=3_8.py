
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 9, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(9, 13, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * (-0.1644084823241768)
        v9 = self.conv3(v8)
        v10 = v7 * (-0.5327659352274278)
        v11 = v10 * 0.6721125975135153
        v12 = v10 * 0.874977282546978
        v13 = v12 * (-0.18878651254249724)
        v14 = self.conv(v13)
        v15 = v14 * (-0.39628569548551407)
        v16 = v14 * (-1.640133269690416)
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 61, 41)
