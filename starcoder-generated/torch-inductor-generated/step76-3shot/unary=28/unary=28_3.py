
class Model(torch.nn.Module):
    def __init__(self, max_value=0.5):
        super().__init__()
        # The minimum and maximum values must satisfy the following conditions:
        # - max_value >= min_value
        # - 0 < max_value - min_value <= 1
        if max_value <= 0 or max_value - 1 <= 0:
            raise ValueError('max_value must be greater than 0 and no less than 1')
        self._max_value = max_value
 
    def forward(self, x1):
        # Input: x1.shape = [N0, N1, N2, N3]
        # Output: x2.shape = [N0, N1, N2, N3]
        x2 = self.linear(x1)
        assert (0 < self._max_value - self._min_value <= 1), 'Incorrect max_value.'
        x3 = torch.clamp_min(x2, min=self._min_value)
        x4 = torch.clamp_max(x3, max=self._max_value)
        return x4

# Initializing the model
m = Model(max_value=0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
