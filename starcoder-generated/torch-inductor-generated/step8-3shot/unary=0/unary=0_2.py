
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v3 + v4
        v13 = v12 * 0.5
        v14 = v12 * v12
        v15 = v14 * v12
        v16 = v15 * 0.044715
        v17 = v12 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v13 * v20
        return v21
# Inputs to the model
x2 = torch.randn(1, 64, 16, 16)
