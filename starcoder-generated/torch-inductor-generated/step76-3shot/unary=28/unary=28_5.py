
class M(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3,3)

    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = torch.clamp_min(t1, min_value=-1)
        return torch.clamp_max(t2, max_value=1)

# Initializing the module
m = M(min_value=-1, max_value=1)

# Input to the model
x1 = torch.randn(1,3)

# Output of the model is x1 + 5.
