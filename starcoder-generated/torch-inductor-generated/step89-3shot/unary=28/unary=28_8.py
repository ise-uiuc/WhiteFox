
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x, min_value = None, max_value = None):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model(min_value=0.0, max_value=1.0)

# Inputs to the model
x = torch.randn(1, 10)
