
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = torch.nn.Linear(1000, 4096)
        self.dense_2 = torch.nn.Linear(4096, 4096)
        self.dense_3 = torch.nn.Linear(4096, 4096)
    def forward(self, x1):
        v1 = self.dense_1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.dense_2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.dense_3(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 1000)
