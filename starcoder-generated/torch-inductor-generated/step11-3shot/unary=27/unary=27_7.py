
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=1)
        self.mul = torch.mul
        self.clamp = torch.clamp
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.mul(v1, v1)
        v3 = self.clamp(v2, self.min_value, self.max_value)
        return v3
min_value = -5
max_value = -2
# Inputs to the model
x1 = torch.randn(1, 1, 22, 49)
