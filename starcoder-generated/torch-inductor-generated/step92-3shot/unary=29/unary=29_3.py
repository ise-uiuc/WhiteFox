
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1195, max_value=2.1147):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 128, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v4 = torch.clamp_max(v2, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 32, 15)
