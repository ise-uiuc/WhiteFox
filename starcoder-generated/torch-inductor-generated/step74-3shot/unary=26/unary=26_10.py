
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 31, 1, stride=1, padding=0, dilation=2, groups=4, bias=True)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 < 3.32
        y3 = y1 * -0.674
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.leaky_relu(y4)
# Inputs to the model
x = torch.randn(1, 2, 93, 77)
