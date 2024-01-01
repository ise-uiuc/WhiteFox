
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Parameter(torch.randn(32768, requires_grad=True))
        self.conv_t = torch.nn.ConvTranspose1d(6, 11, 1, stride=1, padding=0, bias=False)
        self.conv_t.requires_grad = False
    def forward(self, x14):
        v0 = self.input
        l1 = torch.tanh(self.conv_t(x14))
        l2 = l1 > 0
        l3 = l1 * -0.4750
        l4 = torch.where(l2, l1, l3)
        j4 = torch.clamp(v0 * l4, min=-1.0, max=1.0)
        return j4
# Inputs to the model
x14 = torch.randn(123, 6)
