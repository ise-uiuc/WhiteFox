
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.301):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 32, 14, stride=20, padding=4)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, -1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 255, 255)
