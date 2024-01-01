
class Model(torch.nn.Module):
    def __init__(self, min_value=3, max_value=5):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=3)
        v3 = torch.clamp_max(v2, max_value=5)
        return v3

# Initializing the model
# The minimum and maximum values for the clamped value provided here are different from the provided values of the pattern.
m1 = Model(min_value=1, max_value=2)

# Inputs to the model
x1 = torch.randn(1, 8)
