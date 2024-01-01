
class Model(torch.nn.Module):
    def __init__(self, min_value=-1):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
