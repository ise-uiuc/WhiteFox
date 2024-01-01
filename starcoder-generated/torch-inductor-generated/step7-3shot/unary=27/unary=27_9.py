
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=0.01):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 5)
        self.conv2 = torch.nn.Conv2d(3, 2, 5)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv1(torch.clamp_min(x, self.min_value))
        v2 = self.conv2(torch.clamp_max(v1, self.max_value))
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
