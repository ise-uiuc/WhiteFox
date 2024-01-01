
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.2, max_value=3.6):
        super().__init__()
        self.clamp = torch.nn.ReLU6()
        self.add = torch.add
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.clamp(v3)
        v5 = v4 + self.min_value
        v6 = self.clamp(v5)
        v7 = v6 + self.min_value
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
