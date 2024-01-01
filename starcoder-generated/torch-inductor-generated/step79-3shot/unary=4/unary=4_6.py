
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v4 = torch.erf(v2 * 0.7071067811865476)
        v7 = v4 + 1
        v8 = v2 * 0.5
        v11 = v7 * v8
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 5)
