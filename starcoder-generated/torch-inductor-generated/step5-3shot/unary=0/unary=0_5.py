
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(2, 7, 5, stride=1, padding=2)
        self.conv_2 = torch.nn.Conv2d(7, 10, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(10, 15, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv_2(v10)
        v13 = v1 * v10
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v1 * v19
        v21 = v10 + v20
        v23 = v1 + v19
        v26 = v11 * v8
        v28 = v19 * torch.tanh(v11)
        v24 = v11 * v8
        v27 = v11 * v9
        v25 = v28 + v19
        v22 = v24 + v27
        return v21, v22, v23
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
