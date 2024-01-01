
class Model(torch.nn.Module):
    def __init__(self, min_value=0.05, max_value=3000):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, kernel_size=1, stride=1, groups=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
