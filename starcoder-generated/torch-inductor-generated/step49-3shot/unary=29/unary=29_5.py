
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.2442, max_value=-1.029):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 3, 1, stride=2, padding=2, output_padding=2, bias=True, dilation=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 3, 1, stride=(4, 1), padding=(3, 4), dilation=(2, 2), transposed=True, output_padding=(5, 2))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 5, 1, stride=4, padding=0, dilation=1, output_padding=4, groups=1, bias=True, padding_mode='zeros')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(x1)
        v3 = self.conv_transpose3(v2)
        return v1 - v3
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
