
class Model(torch.nn.Module):
    def __init__(self, min_value=0.01, max_value=0.02):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
