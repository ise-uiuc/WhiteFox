
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(2, 2, 2, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(2, 2, (2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.conv_transpose_3 = torch.nn.ConvTranspose3d(2, 2, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False)
    def forward(self, input):
        x1 = input
        x2 = self.conv_transpose_1(x1)
        x3 = self.conv_transpose_2(x1)
        x4 = self.conv_transpose_3(x1)
        return x2 + x3 + x4
# Inputs to the model
input = torch.tensor(((1., 2.), (3., 4.)), requires_grad=True)
