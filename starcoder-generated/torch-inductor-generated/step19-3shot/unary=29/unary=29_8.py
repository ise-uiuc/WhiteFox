
class Model(torch.nn.Module):
    def __init__(self, min_value=5.1917082587859316, max_value=-5.181125051492283):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
        self.hardtanh = torch.nn.Hardtanh(inplace=False)
        self.conv2d = torch.nn.Conv2d(8, 3, 1, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.hardtanh(v3)
        v5 = self.conv2d(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 64)
