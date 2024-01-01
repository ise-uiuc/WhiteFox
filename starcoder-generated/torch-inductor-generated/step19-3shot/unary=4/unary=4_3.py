
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 7)
 
    def forward(self, x2):
        v2 = x2
        v4 = v2 * 0.5
        v6 = v4 * 0.7071067811865476
        v7 = torch.erf(v6)
        v9 = v7 + 1
        v10 = v9 * v4
        v11 = torch.add(v1, v8)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(19, 13)
