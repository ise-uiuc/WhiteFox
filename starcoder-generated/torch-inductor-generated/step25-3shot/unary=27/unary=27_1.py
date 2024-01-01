
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 2)
        self.relu6 = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = torch.clamp_min(v0, self.min_value)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = self.relu6(v2)
        return v3
min_value = 0.003
max_value = 0.151
# Inputs to the model
x1 = torch.randn(1, 3, 172, 172)
