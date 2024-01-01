
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(18, 18, 4, stride=3, padding=4)
        self.conv5 = torch.nn.Conv2d(18, 20, 4, stride=5, padding=2)
        self.conv3 = torch.nn.Conv2d(20, 24, 3, stride=9, padding=8)
    def forward(self, x73):
        v1 = self.conv1(x73)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv5(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        v21 = self.conv3(v20)
        v22 = v21 * 0.5
        v23 = v21 * v21
        v24 = v23 * v21
        v25 = v24 * 0.044715
        v26 = v21 + v25
        v27 = v26 * 0.7978845608028654
        v28 = torch.tanh(v27)
        v29 = v28 + 1
        v30 = v22 * v29
        v31 = torch.add(v10, v30, alpha=1)
        v32 = torch.add(v31, v20, alpha=1)
        return v32
# Inputs to the model
x73 = torch.randn(1, 18, 27, 23)
