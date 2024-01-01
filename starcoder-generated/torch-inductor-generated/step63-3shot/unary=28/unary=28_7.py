
class Model(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.linear = torch.nn.Linear(3, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_val)
        v3 = torch.clamp_max(v2, max_val)
        return v3

# Initializing the model
m = Model(-1.0, 1.0)

# Inputs to the model
x1 = torch.randn(1, 3)
