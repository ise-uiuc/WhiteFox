
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear((-1, 1), 1, bias=False)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        v1 = self.linear(x1)
        v1_safe = v1.clamp(self.min_value, self.max_value)
        return v1_safe

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 256)
