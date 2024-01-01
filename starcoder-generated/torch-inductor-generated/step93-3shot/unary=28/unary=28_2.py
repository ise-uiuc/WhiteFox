
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_min(torch.clamp_max(v1, max_value), min_value)

# Initializing the model
min_value = -0.5
max_value = 0.5
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3)
