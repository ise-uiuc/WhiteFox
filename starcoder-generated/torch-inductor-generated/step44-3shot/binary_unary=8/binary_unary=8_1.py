
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=2, groups=16)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = v2 + v2
        v4 = v3 + v3
        v5 = v2 + v4
        v6 = v2 + v3 + v4 + v2 + v3 + v4 + v5 + v5
        v7 = torch.relu(v6)
        v8 = self.conv1(v7)
        v9 = self.conv1(v8)
        v10 = self.conv1(v9)
        v11 = torch.relu(v10)
        v12 = v7 + v8 + v9
        v13 = v10 + v11
        v14 = v11 + v10
        v15 = torch.relu(v13)
        w5 = v12 + v13
        w6 = v14 + v13 + v13
        w7 = v13 + v15
        w8 = v14 + v15
        w9 = v12 + v14
        torch.cat([ (w5 + w6), w7, w8, w9 ], axis = 1)
        return w5
# Inputs to the model
x1 = torch.randn(1, 16, 28, 32)
