
class Model(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=0, output_padding=0, groups=16, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1.contiguous()
# Inputs to the model
x1 = torch.randn(1, 32, 8, 8)
