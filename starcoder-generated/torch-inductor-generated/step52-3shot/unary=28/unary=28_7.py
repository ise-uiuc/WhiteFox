
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model with keyword arguments for the min and max value to clamp to
m = Model(min_value=0.0, max_value=1.0)

# Initializing the model with keyword arguments for the min and max value to clamp to
m = Model(min_value=0.0, max_value=1.0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
