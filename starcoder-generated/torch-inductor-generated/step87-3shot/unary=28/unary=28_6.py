
class Model(torch.nn.Module):
    def __init__(self, **max_min_values):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10, bias=False)
        self.min_value = min_values
        self.max_value = max_values
 
    def forward(self, x1):
        x2 = torch.flatten(x1, 1)
        v1 = self.linear(x2)
        v2 = torch.clamp(v1, min=self.min_value, max=self.max_value)
        return v2

# Initializing the model
m = Model(min_value=-.5, max_value=+.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
