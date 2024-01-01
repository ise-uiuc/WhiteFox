
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x1, min_value=-1.0, max_value=0.0):
        v1 = self.linear(x1)
        return torch.clamp_min(torch.clamp_max(v1, min_value), max_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
