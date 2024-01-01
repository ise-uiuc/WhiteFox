 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
