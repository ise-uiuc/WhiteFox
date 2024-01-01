
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v4 = v3 + x3
        v5 = v4 * 0.25
        v6 = v5 * 0.7071067811865476
        v7 = torch.erf(v6)
        v8 = v7 * 0.3013381366713429
        v9 = v6 + 1
        v10 = v8 * v9
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 8)
