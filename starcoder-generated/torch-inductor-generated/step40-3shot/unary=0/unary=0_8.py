
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 4, 25, stride=3, padding=0, dilation=2, groups=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 5, stride=2, padding=5, dilation=2, groups=2)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv2(v10)
        v12 = v11 * 0.5
        v13 = v1 * v1
        v14 = v3 * v1
        v15 = v4 * 0.044715
        v16 = v1 + v5
        v17 = v6 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v10 + v13
        v21 = v16 * 0.7978845608028654
        v22 = torch.tanh(v21)
        v23 = v22 + 1
        v24 = v10 + v14
        v25 = v16 * 0.7978845608028654
        v26 = torch.tanh(v25)
        v27 = v26 + 1
        v28 = v10 + v15
        v29 = v11 + v22
        v30 = v29 * v29
        v31 = v22 + 1
        v32 = (v10 + v30) * v31
        return v28
# Inputs to the model
x2 = torch.randn(1, 100, 200, 5)
