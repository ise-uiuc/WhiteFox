
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose3d(3, 3, 2, stride=1, padding=0, dilation=1, groups=1, output_padding=0, bias=False, padding_mode='zeros')
    def forward(self, x31):
        z7 = self.conv_t(x31)
        z8 = z7 > 0
        z9 = z7 * -2.8
        z10 = torch.where(z8, z7, z9)
        return torch.nn.functional.max_pool2d(z10, (1, 2), stride=(8, 1), padding=(1, 0), ceil_mode=False, return_indices=False, padding_mode='zeros')
# Inputs to the model
x31 = torch.randn(64, 3, 64, 2)
