
class Model(torch.nn.Module):
    def __init__(self, min_value=False, max_value=False):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv2d = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0, dilation=1)
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x):
        y = torch.clamp_min(x, self.min_value)
        y = torch.clamp_max(y, self.max_value)
        y = self.conv2d(y)
        y = self.relu(y)
        return y
# Inputs to the model
x = torch.randn(1, 4, 513, 512)
