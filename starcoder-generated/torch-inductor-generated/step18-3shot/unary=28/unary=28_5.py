
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x, min_value, max_value):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
min_value = 0
max_value = 1.0
m = Model(min_value=min_value, max_value=max_value)

# Inputs to the model
x = torch.randn(1, 16)
