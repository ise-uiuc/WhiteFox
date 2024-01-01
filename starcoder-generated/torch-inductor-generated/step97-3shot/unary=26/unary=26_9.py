
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(108, 49, 2, stride=(1, 2, 1), bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.155
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.relu(x2), (14, 185))
# Inputs to the mo
x = torch.randn(2, 108, 46, 19, 40)
