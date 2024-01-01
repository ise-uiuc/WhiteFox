
negative_slope = 0.279
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(10, 10, 2, stride=1, padding=0)
        self.conv2d = torch.nn.Conv2d(10, 10, 1, stride=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_transpose2d(x)
        t2 = self.conv2d(t1)
        t3 = t2 > 0.0
        t4 = t2 * self.negative_slope
        t5 = torch.where(t3, t2, t4)
        return t5
# Inputs to the model
x = torch.randn(1, 10, 4, 4)
