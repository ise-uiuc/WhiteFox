
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 20, kernel_size=3, stride=7, dilation=2, padding=2, output_padding=7)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 4, 2, 2)
