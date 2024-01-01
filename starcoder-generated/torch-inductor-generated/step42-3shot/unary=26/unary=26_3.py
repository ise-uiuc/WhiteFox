
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose3d(65, 1, 2, stride=2, dilation=1, bias=False, output_padding=0, padding=0)
    def forward(self, x8):
        y5 = torch.ones(x8.shape[2:])
        y6 = y5 + 0.562
        y7 = torch.clamp(y6, min=0.5, max=1)
        y8 = torch.ones(x8.shape).cuda()
        y9 = y7 * y8
        y10 = (x8 * y5 * y9)
        return y10
# Inputs to the model
x8 = torch.randn(9, 65, 7, 9, 9).to('cuda')
