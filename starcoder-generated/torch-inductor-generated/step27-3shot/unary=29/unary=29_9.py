
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.5, max_value=-0.1):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0)
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = torch.clamp_min(v2, self.min_value)
        return v3
# Inputs to the model
x2 = torch.randn(1, 8, 7, 7)
