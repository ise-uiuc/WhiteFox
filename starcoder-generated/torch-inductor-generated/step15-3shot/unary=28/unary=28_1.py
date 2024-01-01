
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_max(torch.clamp_min(v1, min_value), max_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
