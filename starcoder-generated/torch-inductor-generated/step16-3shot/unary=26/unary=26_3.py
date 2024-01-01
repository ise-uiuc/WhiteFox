
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 48, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.ConvTranspose2d(48, 128, 3, stride=2, padding=1, dilation=1, bias=True)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = t2 > 0
        t4 = t2 * self.negative_slope
        t5 = torch.where(t3, t2, t4)
        return t5
negative_slope = -0.1
# Inputs to the model
x1 = torch.randn(32, 128, 25, 25)
