
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x2):
        v0 = torch.add(x2, 0.5, 1)
        v1 = torch.mul(v0, 2.0, 0)
        v2 = torch.add(v1, 0.5, 1)
        v3 = v2 * 9.578451291828878e-05
        v4 = v3 / 16
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
