
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_min(torch.clamp_max(v1, kwargs['min_value']), kwargs['max_value'])

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
