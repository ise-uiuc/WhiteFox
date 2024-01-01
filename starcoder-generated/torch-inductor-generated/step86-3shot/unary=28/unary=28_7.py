
class Model(torch.nn.Module):
    def __init__(self, _min=10, _max=22):
        super().__init__()
        self._min = _min
        self._max = _max
 
    def forward(self, x1):
        v1 = x1.permute(0, 2, 3, 1)
        v2 = v1.flatten(0, 1)
        v3 = self.linear(v2)
        v4 = torch.clamp_min(v3, self._min)
        v5 = torch.clamp_max(v4, self._max)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 2, 2, 3)
