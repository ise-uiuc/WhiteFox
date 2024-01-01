
class Model(torch.nn.Module):
    def __init__(self, max_value=0.1):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 3, stride=5, padding=1, bias=False)
        self.max_value = max_value
    def forward(self, x1):
        v1 = torch.clamp_max(self.conv_transpose(x1), self.max_value)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 58, 25)
