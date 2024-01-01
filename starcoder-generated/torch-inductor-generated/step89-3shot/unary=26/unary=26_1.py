
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=2, output_padding=1)
    def forward(self, x0):
        o1 = self.conv_t(x0)
        o2 = o1 > 0
        o3 = o1 * -0.12
        o4 = torch.where(o2, o1, o3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.LeakyReLU()(o4), (1, 1))
# Inputs to the model
x0 = torch.randn(3, 2, 8, 7)
