
class Model(torch.nn.Module):
    def __init__(self, minValue=-0.5, maxValue=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(12, 16)
        self.min_value = minValue
        self.max_value = maxValue

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 12)
