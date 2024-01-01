
class Model(torch.nn.Module):
    def __init__(self, min_value=-10.3456789, max_value=-10.3455555):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=-10.3456789)
        v3 = torch.clamp_max(v2, max_value=-10.3455555)
        return v3

# Initializing the model
m = Model(min_value=-10.3456789, max_value=-10.3455555)

# Inputs to the model
x1 = torch.randn(1, 16)
