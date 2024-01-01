
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(352, 512, (11, 20), stride=1, padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(512, 389, 7, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(389, 273, (6, 13), stride=1, padding=(1, 0))
        self.conv4 = torch.nn.Conv2d(273, 165, (13, 12), stride=1, padding=(1, 3))
        self.conv5 = torch.nn.Conv2d(166, 93, (8, 15), stride=1, padding=(3, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
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
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = self.conv5(v19)
        return v20
# Inputs to the model
x1 = torch.randn(1, 352, 236, 432)
