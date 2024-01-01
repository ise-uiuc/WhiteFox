
class Model(torch.nn.Module):
    def __init__(self, min_value : float = 0., max_value : float = 1.):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 5, 1)
