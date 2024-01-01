
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=6.0):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        z1 = self.linear(x1)
        z2 = torch.clamp_min(z1, self.min_value)
        z3 = torch.clamp_max(z2, self.max_value)
        return z3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
