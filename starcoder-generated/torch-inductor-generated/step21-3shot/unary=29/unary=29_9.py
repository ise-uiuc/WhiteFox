
class Model(torch.nn.Module):
    def __init__(self, min_value=100, max_value=1000):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 16, 2, stride=2, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = torch.rand(1, 1, 5, 5)
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
