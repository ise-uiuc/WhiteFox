
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 14, 3, stride=2, dilation=1, output_padding=(1, 1), groups=1)
        self.convtranspose3 = torchvision.ops.DeformConv2d(14, 22, 2, stride=1, padding=1)
        self.convtranspose4 = torchvision.ops.deform_conv2d(x1, self.convtranspose3.weight, self.convtranspose3.offset, 1, 1, 1, 1, 1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.convtranspose3(v2)
        v4 = torch.tanh(v4)
        v5 = self.convtranspose4(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 11, 11)
