
class Model(torch.nn.Module):
    def __init__(self, min_value=0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, 2, stride=2, padding=0)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
