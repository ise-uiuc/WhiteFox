
class Model(torch.nn.Module):
    def __init__(self, min_value=-6.2, max_value=-5.5):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v1.retain_grad()
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
