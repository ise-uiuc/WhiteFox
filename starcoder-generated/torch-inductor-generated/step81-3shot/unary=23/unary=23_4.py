
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(21, 6, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(6, 21, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        y = torch.tanh(self.conv2(x))
        z = self.conv(y)
        return z
# Inputs to the model
x1 = torch.randn(1, 21, 24, 24)
