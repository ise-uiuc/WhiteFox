
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=1.)
        v3 = torch.clamp_max(v2, max_value=2.)
        return v3

# Initializing the model with the required keyword arguments
m = Model(min_value=1., max_value=2.)

# Inputs to the model
x1 = torch.randn(1, 3)
