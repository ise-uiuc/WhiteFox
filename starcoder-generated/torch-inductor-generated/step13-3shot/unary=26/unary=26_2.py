
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Conv2d(3, 16, 2, stride=1, groups=2, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 1.34
        x4 = torch.where(x2, x1, x3)
        return x1
# Inputs to the model
x = torch.randn(5, 3, 16, 16)
