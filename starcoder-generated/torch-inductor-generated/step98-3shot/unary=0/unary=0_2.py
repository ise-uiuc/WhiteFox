
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(41, 33, 1, stride=1, padding=12)
        self.conv2 = torch.nn.Conv2d(9, 36, 1, stride=1, padding=37)
    def forward(self, x08, x29, x12):
        v1 = self.conv1(x08)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = x29 + v10
        v11 = self.conv2(v10)
        v12 = v11 * 0.5
        v13 = v1 + v12
        v14 = v13 * v3
        v15 = v11 * v1
        v16 = v15 + v11
        v17 = v14 + v16
        v18 = v17 * 0.044715
        v19 = v17 + v18
        v20 = v19 * 0.7978845608028654
        v21 = x12 + v20
        v22 = v21 * 0.5
        v23 = torch.tanh(v22)
        v24 = v23 + 1
        v25 = v24 + v2
        v26 = v25 * v9
        return v26    
# Inputs to the model
x08 = torch.randn(1, 41, 8, 33)
x29 = torch.randn(1, 9, 21, 6)
x12 = torch.randn(1, 1, 23, 23)
