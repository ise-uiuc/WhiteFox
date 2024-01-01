
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
     
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=min_value)
        return torch.clamp_max(v2, max_value=max_value)

# Input to the model
x1 = torch.randn(1, 64)

# Initializing the model
m = Model(0.5, 0.7071067811865476)
