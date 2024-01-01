
class Model(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self._min_value = min_value
        self._max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self._min_value)
        v3 = torch.clamp_max(v2, self._max_value)
        return v3

# Initializing the model
min_value = 0.5
max_value = 0.9
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3)
