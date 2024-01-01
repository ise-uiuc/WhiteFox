
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1024)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = x1 * 0.0009765625
x3 = x1 * 0.0009711390513643345
x4 = (v3 + 1e-38)
