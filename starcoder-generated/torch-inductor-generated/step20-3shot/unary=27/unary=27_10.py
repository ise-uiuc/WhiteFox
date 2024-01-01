
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.125, alpha=0.2):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.activation = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3, padding=1),
            torch.nn.ReLU()
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v1b = v1
        v2 = self.activation(v1)
        v3 = torch.clamp_max(v1b + self.alpha * v2, self.max_value)
        v4 = torch.clamp_min(v3, self.max_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
