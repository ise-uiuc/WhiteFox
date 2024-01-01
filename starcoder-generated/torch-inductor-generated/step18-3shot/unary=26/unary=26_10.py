
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 12, 2, stride=2, groups = 12)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2.flatten()
        x4 = x3 > 0
        x5 = x3 * 0.5
        x6 = torch.where(x4, x3, x5)
        x7 = x6.reshape(x6.size(0), 12, 12, 12)
        return x7 + torch.nn.functional.adaptive_avg_pool2d(x7, (2, 2)) + torch.nn.functional.unfold(x7, kernel_size = 2, stride = 2, padding = 0)
# Inputs to the model
x1 = torch.randn(4, 480, 10, 10)
