
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(41, 79, 3, stride=1, padding=0, output_padding=0, dilation=2, groups=1, bias=True, padding_mode='zeros', device=device, dtype=torch.float16)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * -0.41
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(8, 41, 9, 8)
