
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = x1 * 0.5
        v2 = x1 * x1
        v3 = v2 * x1
        v4 = v3 * 0.044715
        v5 = x1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v1 * v8
        v10 = x2 * 0.5
        v11 = x2 * x2
        v12 = v11 * x2
        v13 = v12 * 0.044715
        v14 = x2 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v10 * v17
        v19 = x3 * 0.5
        v20 = x3 * x3
        v21 = v20 * x3
        v22 = v21 * 0.044715
        v23 = x3 + v22
        v24 = v23 * 0.7978845608028654
        v25 = torch.tanh(v24)
        v26 = v25 + 1
        v27 = v19 * v26
        v28 = x4 * 0.5
        v29 = x4 * x4
        v30 = v29 * x4
        v31 = v30 * 0.044715
        v32 = x4 + v31
        v33 = v32 * 0.7978845608028654
        v34 = torch.tanh(v33)
        v35 = v34 + 1
        v36 = v28 * v35
        v37 = v9 + v18 + v27 + v36
        v38 = v37 * 0.7978845608028654
        v39 = torch.tanh(v38)
        v40 = v39 + 1
        return v40
x1 = torch.randn(1, 128, 100, 120)
x2 = torch.randn(1, 128, 100, 120)
x3 = torch.randn(1, 128, 100, 120)
x4 = torch.randn(1, 128, 100, 120)
