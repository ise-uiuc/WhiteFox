
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(96, 16)
        self.linear2 = torch.nn.Linear(16, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, 10)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.tanh(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.linear2(v12)
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = torch.tanh(v18)
        v20 = self.linear3(v19)
        v21 = v20 * 0.5
        v22 = v20 * 0.7071067811865476
        v23 = torch.erf(v22)
        v24 = v23 + 1
        v25 = v21 * v24
        return self.linear4(v25)
# Inputs to the model
x1 = torch.randn(1, 96)
