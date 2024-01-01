
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.m = torch.nn.Linear(3, 4)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v = self.m(x)
        return torch.clamp_max(torch.clamp_min(v, self.min_value), self.max_value)

# Initializing the model
m = Model(min_value=0, max_value=1)

# Inputs to the model
x = torch.randn(1, 3)
