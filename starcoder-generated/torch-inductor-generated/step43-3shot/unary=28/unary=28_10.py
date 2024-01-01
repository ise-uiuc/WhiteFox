
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.linear = torch.nn.Linear(in_features=16384, out_features=10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(0, 1)

# Inputs to the model (x1 does not need to be of size (1, 16384)
x1 = torch.randn(1, 16384)
