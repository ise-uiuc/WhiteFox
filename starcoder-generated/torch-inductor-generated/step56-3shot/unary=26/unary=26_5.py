
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(20, 37, 15, stride=2, padding=4, bias=False, dilation=2)
    def forward(self, x1):
        a1 = self.conv_t(x1)
        a2 = a1 > 0
        a3 = a1 * -49.7
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x1 = torch.randn(21, 20, 11, device='cuda')
