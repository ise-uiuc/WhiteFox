
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=5):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 1, stride=1, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 7, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = self.conv_transpose1(v2)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 56, 71)
