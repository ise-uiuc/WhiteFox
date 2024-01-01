
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model with appropriate values for min_value and max_value
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 32, 2, 2)
