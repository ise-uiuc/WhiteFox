
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.min_value = min_value
        self.max_value = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
min_value = -0.5
max_value = 0.5
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3)
