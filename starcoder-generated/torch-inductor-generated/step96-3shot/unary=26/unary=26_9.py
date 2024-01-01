
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 14, 8, 2, 4, output_padding=4, groups=16, bias=False)
    def forward(self, x):
        x5 = self.conv_t(x)
        x6 = x5 > 0
        x7 = x5 * neg_slope
        x8 = torch.where(x6, x5, x7)
        return torch.nn.functional.interpolate(torch.nn.LeakyReLU(inplace=True)(torch.nn.functional.gelu(torch.nn.functional.dropout(x8))), size=[24, x.shape[-2].value])
# Inputs to the model
x = torch.randn(1, 4, 100, 15, device='cuda')
