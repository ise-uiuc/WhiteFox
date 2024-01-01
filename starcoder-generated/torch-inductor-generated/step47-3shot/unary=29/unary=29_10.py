
class Model(torch.nn.Module):
    def __init__(self, min_value=10.0) :
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 12, 1, stride=1, padding=1)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return torch.clamp(v1, min=self.min_value)
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
