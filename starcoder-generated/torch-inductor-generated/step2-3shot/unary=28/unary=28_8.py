
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=2)
        v3 = torch.clamp_max(v2, max_value=3)
        return v3

# Initializing the model
m = Model(min_value=2, max_value=3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
