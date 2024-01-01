
class Model(torch.nn.Module):
    def __init__(self, min_value=0., max_value=1.):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v1 = torch.clamp_min(v1, min_value)
        return torch.clamp_max(v1, max_value)

# Initializing the model
m = Model(min_value=-1., max_value=2.)

# Inputs to the model
x1 = torch.randn(5, 16)
