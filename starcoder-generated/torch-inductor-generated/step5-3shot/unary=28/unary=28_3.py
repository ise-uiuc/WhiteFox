
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return torch.clamp_max(v2, self.max_value)

# Initializing the model
m = Model(-5, 6)

# Inputs to the model
x1 = torch.rand(3, 5)
