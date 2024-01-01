
class Model(torch.nn.Module):
    def __init__(self, min_value=0.2, max_value=5.0):
        super().__init__()
        self.max_unpooling = torch.nn.MaxUnpool2d(1, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 10, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.max_unpooling(v3, x1, 1)
        return v4
# Inputs to the model
x1 = torch.zeros((1, 3, 32, 32), dtype=float)
x2 = torch.randn(1, 9, 32, 32)
