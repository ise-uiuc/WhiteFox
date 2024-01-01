
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.7, max_value=1.6):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 2, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x3 = torch.randn(1, 4, 44, 44)
