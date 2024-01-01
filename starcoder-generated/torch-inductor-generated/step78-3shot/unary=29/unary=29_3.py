
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.358296, max_value=1.76060):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 4, 2, stride=3, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 5, 5)
