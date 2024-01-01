
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0.0)
        v3 = torch.clamp_max(v2, 1.0)
        return v3

# Initializing the model
m = Model()

# Initial values of min_value and max_value
min_value = 1.0
max_value = 2.0

# Inputs to the model
x1 = torch.randn(8)
