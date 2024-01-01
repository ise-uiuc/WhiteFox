
class Model(torch.nn.Module):
    def __init__(self, min_value=5, max_value=5):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.gelu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 4, 1, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.gelu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv_transpose(v3)
        v5 = torch.clamp_min(v4, self.min_value)
        v6 = torch.clamp_max(v5, self.max_value)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
