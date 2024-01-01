
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 3)
