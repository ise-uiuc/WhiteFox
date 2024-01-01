
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, x1, vmin, vmax):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, vmin)
        v3 = torch.clamp_max(v2, vmax)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
min_value = 20.0
max_value = 30.0
