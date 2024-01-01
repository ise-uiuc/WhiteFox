
class Model(torch.nn.Module):
    def __init__(self, min_value=2.3, max_value=7.3):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=-2.1)

# Inputs to the model
x1 = torch.randn(1, 64)
