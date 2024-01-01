
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.4, max_value=-3.5):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
