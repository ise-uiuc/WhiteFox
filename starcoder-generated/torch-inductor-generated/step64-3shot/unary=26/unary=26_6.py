
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(49, 3, 5, stride=1, padding=0, bias=False)
    def forward(self, x):
        o1 = self.conv_t(x)
        o2 = o1 > 0
        o3 = o1 * -2.1502
        o4 = torch.where(o2, o1, o3)
        return o4
# Inputs to the model
x = torch.randn(2, 49)
