
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v6 = clamp(v1, 0.0, 6.0)
        v2 = v1 + v6
        v3 = v2 * v6
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
