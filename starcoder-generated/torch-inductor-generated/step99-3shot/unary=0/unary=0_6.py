
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(56, 18, 3, stride=4, padding=1)
        self.conv2 = torch.nn.Conv2d(18, 54, 1, stride=3, padding=2)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv2(x5)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        v21 = v10 + v20
        v22 = v21 * 0.7978845608028654
        v23 = torch.tanh(v22)
        v24 = v23 + 1
        v25 = v10 * v24
        return v25
# Inputs to the model
x5 = torch.randn(1, 56, 14, 86)
