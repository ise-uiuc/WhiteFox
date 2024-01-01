
class Model(torch.nn.Module):
    def __init__(self, min_value=6.1, max_value=6.2):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 2, 1, stride=1)
        self.conv2d = torch.nn.Conv2d(2, 4, 1, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv2d(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
