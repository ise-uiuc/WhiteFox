
class Model(torch.nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.linear = torch.nn.Linear(n1, n2)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7

# Initializing the model
m = Model(256, 512)

# Inputs to the model
x2 = torch.randn(8, 256)
