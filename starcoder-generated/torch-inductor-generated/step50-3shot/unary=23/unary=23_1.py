
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=0, dilation=1, output_padding=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
