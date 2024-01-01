
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1, output_padding=0)
    def forward(self, x):
        o1 = self.conv_t(x)
        o2 = o1 > 0
        o3 = o1 * -0.4
        o4 = torch.where(o2, o1, o3)
        o5 = torch.nn.functional.avg_pool2d(o4, 8, 2, 1, 1)
        return torch.nn.functional.hardtanh(o5, -6.027711750132898, 6.027711750132898)
# Inputs to the model
x = torch.randn(65, 4, 53, 40)
