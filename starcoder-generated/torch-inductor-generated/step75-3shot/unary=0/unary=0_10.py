
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 4, 2, stride=1, padding=0)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        x1 = v2 * v9
        v10 = self.conv3(x1)
        v11 = v10 * 0.5
        v12 = v10 * v10
        v13 = v12 * v10
        v14 = v13 * 0.044715
        v15 = v10 + v14
        v16 = v15 * 0.7978845608028654
        v17 = torch.tanh(v16)
        v18 = v17 + 1
        v19 = v11 * v18
        return v19
# Inputs to the model
x4 = torch.randn(1, 1, 14, 14)
