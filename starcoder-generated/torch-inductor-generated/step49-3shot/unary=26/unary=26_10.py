
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(623, 154, 6, stride=1, padding=2, bias=True)
    def forward(self, x2):
        k1 = self.conv_t(x2)
        k2 = k1 > 0
        k3 = k1 * -0.608
        k4 = torch.where(k2, k1, k3)
        return torch.nn.functional.adaptive_avg_pool2d(k4, (1, 1))
# Inputs to the model
x2 = torch.randn(6, 623, 73, 36)
