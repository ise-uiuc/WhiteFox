
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.min = min_value
        self.max = max_value

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3

# Initializing the model
m1 = Model(-10.0, 10.0) # The inputs for this model are all real-valued numbers in [-10, 10).

# Inputs to the model
x1 = torch.randn(1, 3)
