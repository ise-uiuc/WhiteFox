
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 4, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=0, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.nn.functional.elu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
