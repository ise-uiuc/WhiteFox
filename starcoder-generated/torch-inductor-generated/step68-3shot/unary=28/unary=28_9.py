
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=6.0):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.min_value)
        return torch.clamp(v2, self.min_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
