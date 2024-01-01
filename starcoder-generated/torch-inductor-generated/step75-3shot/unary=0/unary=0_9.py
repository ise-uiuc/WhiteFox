
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 13, 2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(13, 13, 3, stride=1, padding=1)
    def forward(self, x34444):
        v1 = self.conv(x34444)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = torch.clone(v10)
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v13 * v12
        v15 = self.conv2(v14)
        v16 = v15 * 0.5
        v17 = v15 * v15
        v18 = v17 * v15
        v19 = v18 * 0.044715
        v20 = v15 + v19
        v21 = v20 * 0.7978845608028654
        v22 = torch.tanh(v21)
        v23 = v22 + 1
        v24 = v16 * v23
        v25 = v10 + v24
        return v25
# Inputs to the model
x34444 = torch.randn(1, 1, 33, 12)
