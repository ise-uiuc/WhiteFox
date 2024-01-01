
class Model(torch.nn.Module):
    def __init__(self, min_value=-9.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(15, 68, 4, stride=2, padding=2)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, 1.0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 15, 33, 38, 64)
