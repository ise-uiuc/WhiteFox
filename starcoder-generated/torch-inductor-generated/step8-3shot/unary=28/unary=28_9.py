
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._min_value = kwargs.get("constant", 0)
        self._max_value = kwargs.get("constant", 6)
        self.linear = torch.nn.Linear(100, 200)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self._min_value)
        v3 = torch.clamp_max(v2, self._max_value)
        return v3

# Initializing the model
m = Model(constant=0.0001)

# Inputs to the model
x1 = torch.randn(1, 100)
