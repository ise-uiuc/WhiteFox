
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v1 + v7
        v9 = v8 * 0.1
        v10 = v1 + v7
        v11 = v10 * 0.3
        v12 = v10 * 0.5
        v13 = torch.tanh(v12)
        v14 = v13 * 0.5
        v15 = v10 + v14
        v16 = v10 + v15
        v17 = v10 + v15
        v18 = v17 * 0.1
        v19 = torch.mul(v13, v18)
        v20 = torch.exp(v19)
        return v20
# Inputs to the model
x1 = torch.randn(1, 8, 112, 112)
