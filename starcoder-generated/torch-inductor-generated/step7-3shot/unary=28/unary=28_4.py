
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self._min_value = min_value
        self._max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self._min_value)
        v3 = torch.clamp_max(v2, self._max_value)
        return v3

# Initializing the model
min_val = 0.5
max_val = 1.5
m = Model(min_val, max_val)

# Inputs to the model
x1 = torch.randn(1, 3)
