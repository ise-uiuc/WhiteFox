
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 1, 4, stride=4, padding=1)
    def forward(self, x10):
        x12 = self.conv_t(x10)
        x13 = x12 > 0
        x14 = x12 * -0.515
        x15 = torch.where(x13, x12, x14)
        x16 = torch.nn.functional.max_pool2d(x15, stride=1, kernel_size=(3, 9), padding=(1, 0))
        x17 = torch.nn.functional.adaptive_avg_pool2d(x16, (1, 1))
        return torch.nn.functional.adaptive_avg_pool2d(x17, (1, 1))
# Inputs to the model
x10 = torch.randn(15, 8, 7, 9)
