
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(6, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear(torch.cat([x1, x2], dim=1))
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(0.2, 0.8)

# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 6)
