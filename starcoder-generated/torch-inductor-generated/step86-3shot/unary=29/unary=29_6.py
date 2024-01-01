
class Model(torch.nn.Module):
    def __init__(self, min_value=5.6802e-07, max_value=5.7119e-05):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 22, 4, stride=4, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 13, 12)
