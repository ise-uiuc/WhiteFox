
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        v2 = self.lin(x2)
        v4 = v2 * 0.5
        v6 = v2 * 0.7071067811865476
        v8 = torch.erf(v6)
        v10 = v8 + 1
        v12 = v4 * v10
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
