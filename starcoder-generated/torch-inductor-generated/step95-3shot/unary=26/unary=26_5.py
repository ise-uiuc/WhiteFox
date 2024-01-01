
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose1d(334, 418, 1, stride=1, padding=0, bias=True)
    def forward(self, x19):
        u1 = self.conv_t(x19)
        u2 = u1 > 0
        u3 = u1 * -0.162322
        u4 = torch.where(u2, u1, u3)
        return torch.nn.functional.adaptive_avg_pool2d(u4, (7, 5))
# Inputs to the model
x19 = torch.randn(4, 334, 24)
