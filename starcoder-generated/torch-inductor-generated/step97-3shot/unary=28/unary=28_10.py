
class Model(torch.nn.Module):
    def __init__(self, max_value=1.76, min_value=-1.85):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v2

# Initializing the model
m = Model(max_value=1.76, min_value=-1.85)

# Inputs to the model
x1 = torch.randn(1, 16)
