
class Model(torch.nn.Module):
    def __init__(self, min_value=0.6381):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(3.9523)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 17, 1, stride=1, padding=0, dilation=2, output_padding=0)
        self.min_value = min_value
    def forward(self, x9):
        v1 = self.conv_transpose(x9)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
x9 = torch.randn(1, 3, 128, 128)
