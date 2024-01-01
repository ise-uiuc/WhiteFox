
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 7)
        self.min_val = torch.nn.Parameter(torch.tensor(-2.0))
        self.max_val = torch.nn.Parameter(torch.tensor(3.0))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_val)
        v3 = torch.clamp_max(v2, self.max_val)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

# Output of initial model (not an issue)
