
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.131, max_value=16.449):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(3, 1, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x8):
        v5 = self.conv_transpose3d(x8)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        return v7
# Inputs to the model
x8 = torch.randn(1, 3, 19, 19, 19)
