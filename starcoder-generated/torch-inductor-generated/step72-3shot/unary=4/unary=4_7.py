
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = v1 * 0.5
        v4 = v2 * 0.5
        v5 = v1 * 0.7071067811865476
        v6 = v2 * 0.7071067811865476
        v7 = torch.erf(v5)
        v8 = torch.erf(v6)
        v9 = v7 + 1
        v10 = v8 + 1
        v11 = v3 * v9
        v12 = v4 * v10
        return v11 + v12

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
