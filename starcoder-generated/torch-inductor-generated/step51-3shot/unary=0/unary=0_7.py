
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 * 0.5
        v2 = v1 * v1
        v3 = v2 * v1
        v4 = v3 * 0.044715
        v5 = x1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v8 * 0.5
        v10 = v8 * v8
        v11 = v10 * v8
        v12 = v11 * 0.044715
        v13 = v8 + v12
        v14 = v13 * 0.7978845608028654
        v15 = torch.tanh(v14)
        v16 = v15 + 1
        v17 = v9 * v16
        return v17
# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)
