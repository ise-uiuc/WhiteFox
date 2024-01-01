
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        x = self.linear(x1)
        y = torch.clamp_min(x, self.min_value)
        z = torch.clamp_max(y, self.max_value)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
