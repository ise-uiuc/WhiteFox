
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2048, 512, 4, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv_transpose2 = torch.nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, output_padding=1, groups=1, bias=False, dilation=1, padding_mode='zeros')
        self.conv_transpose3 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        self.conv_transpose4 = torch.nn.ConvTranspose2d(64, 1, 8, stride=2, padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
    def forward(self, *args, **kwargs):
        v1 = self.conv_transpose1(*args, **kwargs)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3) + 0.5
        return v4
# Inputs to the model
x1 = torch.randn(1, 2048, 50, 50)
