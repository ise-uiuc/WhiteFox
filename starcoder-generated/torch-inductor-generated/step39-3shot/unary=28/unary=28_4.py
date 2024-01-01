
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()

        self._min_value = min_value
        self._max_value = max_value

    def forward(self, x1):
        t1 = torch.mm(x1, self.weight) + self.bias
        t2 = torch.clamp(t1, self.min_value)
        t3 = torch.clamp(t2, self.max_value)
        return t3

# Initializing the model
args = [1, 3]
kwargs = {
  'min_value': -1.0,
  'max_value': 1.0,
}
m = Model(*args, **kwargs)

# Inputs to the model
x1 = np.random.randn(1, 3)

