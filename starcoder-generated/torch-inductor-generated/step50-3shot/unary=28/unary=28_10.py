
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=32, out_features=64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=0)
        v3 = torch.clamp_max(v2, max_value=0)
        return v3
 
# Initializing with constants
min_value = 0
max_value = 100
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 32)
