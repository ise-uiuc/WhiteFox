
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 240, 1, stride=2, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv1(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 + 1
        v18 = v12 * v17
        v19 = self.conv2(v18)
        v20 = v19 * 0.5
        v21 = v19 * v19
        v22 = v21 * v19
        v23 = v22 * 0.044715
        v24 = v19 + v23
        v25 = v24 + 1
        v26 = v20 * v25
        return v26
# Inputs to the model
x3 = torch.randn(1, 64, 32, 32)
