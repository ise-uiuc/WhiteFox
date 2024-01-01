
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(64, 4, 4, stride=2, padding=1, dilation=1, output_padding=0, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        v1 = self.conv_transpose2d(x)
        return v1
# Inputs to the model
x = torch.randn(1, 64, 56, 56)
