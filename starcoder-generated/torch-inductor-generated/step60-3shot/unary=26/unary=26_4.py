
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 1, 2, stride=1, padding=1, bias=False)
    def forward(self, x):
        x1 = torch.nn.ReLU()(x)
        x2 = self.conv_t(x1)
        x3 = x1 > 0
        x4 = x2 > 0
        x5 = x1 * -0.00563876
        x6 = x2 * -0.00448292
        x7 = torch.where(x3, x2, x5)
        x8 = torch.where(x4, x1, x6)
        return x7, x8
# Inputs to the model
x = torch.randn(7, 19, 23, 30, device='cuda')
