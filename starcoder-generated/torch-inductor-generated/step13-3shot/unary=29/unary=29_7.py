
class Model(torch.nn.Module):
    def __init__(self, min_value=-73, max_value=41):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 5, stride=2, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
