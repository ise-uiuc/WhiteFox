
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(83, 24, 6, stride=4, padding=3, bias=False)
    def forward(self, x1):
        o1 = self.conv_t(x1)
        o2 = o1 > 0
        o3 = o1 * 0.065540657
        o4 = torch.where(o2, o1, o3)
        return o4
# Inputs to the model
x1 = torch.randn(437, 83, 5, 68)
