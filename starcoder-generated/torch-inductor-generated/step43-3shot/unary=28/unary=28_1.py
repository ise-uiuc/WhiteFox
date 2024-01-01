
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model with the minimum and maximum values
m = Model(min_value=-20, max_value=5)

# Initializing the model with an invalid minimum value
m = Model(min_value=10000, max_value=5)

# Initializing the model with an invalid maximum value
m = Model(min_value=-20, max_value=-20000)

# Initializing the model with invalid maximum and minimum values
m = Model(min_value=10000, max_value=-20000)

# Inputs to the model
x1 = torch.randn(1, 8)
