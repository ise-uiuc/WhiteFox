
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose1d(166, 209, 1, stride=1, padding=1, dilation=1, groups=1, bias=True)
    def forward(self, x3):
        h1 = self.conv_t(x3)
        h2 = h1 > 0
        h3 = h1 * 2.125
        h4 = torch.where(h2, h1, h3)
        return torch.nn.functional.softmax(h4, dim=-1)
# Inputs to the model
x3 = torch.randn(2, 166, 29)
