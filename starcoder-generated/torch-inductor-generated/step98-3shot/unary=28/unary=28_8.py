
class Model(torch.nn.Module):
    def __init__(self, min_value=-18.24, max_value=82.796):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp_min(self.min_value)
        v3 = v2.clamp_max(self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
