
class Model(torch.nn.Module):
    def __init__(self, min_value=-3, max_value=3):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.tanh()
        v3 = torch.clamp(v2, min=self.min_value, max=self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
