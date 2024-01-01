
class Model(torch.nn.Module):
    def __init__(self, min_value=-3, max_value=3):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(1, 2, 1, stride=1, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x0):
        v1 = self.conv_transpose3d(x0)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x0 = torch.randn(1, 1, 9, 9, 9)
