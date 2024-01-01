
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=10):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2d_4 = torch.nn.Conv2d(8, 6, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d_4(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
