
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.3, max_value=None):
        super().__init__()
        self.max_value = max_value
        self.linear = torch.nn.Linear(6, 1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6)
