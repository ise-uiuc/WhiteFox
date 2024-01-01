
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv10x10 = torch.nn.Conv2d(1, 1, 10, stride=10, padding=5)
        self.conv20x20 = torch.nn.Conv2d(1, 1, 10, stride=10, padding=5)
        self.conv20x20_ = torch.nn.Conv2d(1, 1, 100, stride=100, padding=50)
        self.conv80x80 = torch.nn.Conv2d(1, 1, (8, 8), stride=8, padding=4)
    def forward(self, x1):
        v1 = self.conv10x10(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv20x20(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v110 = v12 * v19
        v21 = self.conv20x20_(v11)
        v22 = v21 * 0.5
        v23 = v21 * v21
        v24 = v23 * v21
        v25 = v24 * 0.044715
        v26 = v21 + v25
        v27 = v26 * 0.7978845608028654
        v28 = torch.tanh(v27)
        v29 = v28 + 1
        v210 = v22 * v29
        v31 = self.conv80x80(v11)
        v32 = v31 * 0.5
        v33 = v31 * v31
        v34 = v33 * v31
        v35 = v34 * 0.044715
        v36 = v31 + v35
        v37 = v36 * 0.7978845608028654
        v38 = torch.tanh(v37)
        v39 = v38 + 1
        v310 = v32 * v39
        v40 = v10 + v110 + v210 + v310
        return v40
# Inputs to the model
x1 = torch.randn(1, 1, 256, 325)
