
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(48, 38, kernel_size=(2, 5), stride=(1, 2), padding=(1, 2), dilation=(1, 2), groups=4, bias=True, padding_mode='zeros')
        self.conv_transpose2 = torch.nn.ConvTranspose2d(38, 19, kernel_size=(2, 5), stride=(2, 1), padding=(1, 2), dilation=(1, 2), groups=3, bias=True, padding_mode='zeros')
        self.conv_transpose3 = torch.nn.ConvTranspose2d(19, 9, kernel_size=(5, 1), stride=(1, 1), padding=(2, 1), dilation=(2, 1), groups=2, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 48, 27, 22)
