
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = x2 * 0.5
        v3 = v2 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.linear(x2)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = torch.cat([v12, v6], dim=1)
        return v13

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 256)
