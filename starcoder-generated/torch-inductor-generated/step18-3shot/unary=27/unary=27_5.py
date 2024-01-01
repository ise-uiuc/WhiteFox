
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value, kernel_size=2):
        super().__init__()
        self.conv = torch.nn.Conv1d(32, 32, 2, stride=2, padding=0, dilation=0)
        self.min_value = min_value
        self.max_value = max_value
        self.kernel_size = 2
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = 0.1
max_value = 0.1
# Inputs to the model
x1 = torch.randn(1, 32, 100)
