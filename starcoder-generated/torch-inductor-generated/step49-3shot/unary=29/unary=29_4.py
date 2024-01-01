
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.6256, max_value=0.):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x4):
        v1 = self.conv_transpose2d(x4)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x4 = torch.randn(1, 1, 4, 4)
