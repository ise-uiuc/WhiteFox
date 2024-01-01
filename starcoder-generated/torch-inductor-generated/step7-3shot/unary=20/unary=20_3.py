
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(kernel_size=6, stride=2, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=9, stride=1, padding=0, output_padding=2, dilation=1, groups=1, bias=True)
        self.acti = torch.nn.Sigmoid()
    def forward(self, X0):
        v1 = self.conv(X0).flatten(start_dim=1, end_dim=-1)
        v2 = self.conv_t(v1).reshape(1,1,18,30)
        v3 = self.acti(v2)
        return v3
# Inputs
x1 = torch.randn(1, 4, 21, 10)
