
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 5, 5, stride=1)
    def forward(self, x):
        x = self.conv_t(x)
        x1 = x + torch.nn.functional.adaptive_avg_pool2d(x * 0.303, (1, 1))
        x2 = x1 > 1
        x4 = torch.where(x2, x, x1)
        x5 = x4 > 1
        return torch.where((x5 & ~x2) ^ x4, x4, x5)
# Inputs to the model
x = torch.randn(5, 2, 5, 5)
