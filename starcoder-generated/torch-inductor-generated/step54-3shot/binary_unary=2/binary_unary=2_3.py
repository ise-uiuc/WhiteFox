
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features16 = conv2d.Conv2d(3, 16, 7, stride=2, padding=3, bias=False)
        self.features32 = conv2d.Conv2d(16, 32, 5, stride=2, padding=2, bias=False)
        self.features32_2 = conv2d.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.features64_1 = conv2d.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.features64_2 = conv2d.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
    def forward(self, x):
        x16 = self.features16(x)
        x32 = self.features32(x16)
        x32_2 = self.features32_2(x32)
        y = self.features64_1(x32_2)
        y = self.features64_2(x32_2)
        return y
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
