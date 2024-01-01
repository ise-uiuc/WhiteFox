
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min_value)
        return torch.clamp_max(v2, max_value=self.max_value)

# Initializing the model
m = Model()

# Changing the keyword arguments
m.min_value = torch.randn(3, 1)
m.max_value = torch.randn(3)

# Inputs to the model
x1 = torch.randn(3, 5)
