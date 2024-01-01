
class Model(torch.nn.Module):
    def __init__(self, max_value=189.91000366210938):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=4, padding=1)
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, -0.0)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 65, 255)
