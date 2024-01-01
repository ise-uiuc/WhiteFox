
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(51, 264, 1, stride=1, padding=0, bias=True)
    def forward(self, x18):
        l1 = self.conv_t(x18)
        l2 = l1 > 0
        l3 = l1 * -3.3682
        l4 = torch.where(l2, l1, l3)
        return torch.nn.functional.adaptive_avg_pool2d(l4, (1, 1))
# Inputs to the model
x18 = torch.randn(11, 51, 13, 24)
