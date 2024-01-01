
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        v1 = v0 * 0.5
        v2 = v0 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 2)
