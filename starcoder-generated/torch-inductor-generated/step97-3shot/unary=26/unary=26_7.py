
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 6, 1, stride=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.00449
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool2d(x4, (1, 1))
# Inputs to the model
x = torch.randn(1, 7, 7, 7)
