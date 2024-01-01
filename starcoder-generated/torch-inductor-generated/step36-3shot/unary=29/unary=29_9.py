
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=631):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(10, 1, padding=10)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 192, 811)
