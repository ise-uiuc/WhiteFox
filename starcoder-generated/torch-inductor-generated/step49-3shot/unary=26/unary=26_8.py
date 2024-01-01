
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(369, 223, 5, stride=2, padding=2, bias=True)
    def forward(self, x11):
        o1 = self.conv_t(x11)
        o2 = o1 > 0
        o3 = o1 * -0.4190
        o4 = torch.where(o2, o1, o3)
        return torch.nn.functional.adaptive_avg_pool2d(o4, (1, 1))
# Inputs to the model
x11 = torch.randn(25, 369, 46, 60)
