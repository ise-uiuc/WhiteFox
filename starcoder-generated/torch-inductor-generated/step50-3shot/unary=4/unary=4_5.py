
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(29, 10)
 
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), 1)
        v2 = self.linear(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9)
x2 = torch.randn(1, 19)
