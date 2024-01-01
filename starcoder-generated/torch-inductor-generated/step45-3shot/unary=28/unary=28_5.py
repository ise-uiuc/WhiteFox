
class Model2(torch.nn.Module):
    def __init__(self, _min, _max):
        super().__init__()
        self._min, self._max = 0, 1
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self._min)
        v3 = torch.clamp_max(v2, self._max)
        return v3

# Initializing the model
m = Model2(float('-inf'), float('inf'))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
