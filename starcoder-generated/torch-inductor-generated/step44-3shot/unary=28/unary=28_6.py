
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model(min_value=-2, max_value=2)

# Inputs to the model
x1 = torch.randn(128, 10)
