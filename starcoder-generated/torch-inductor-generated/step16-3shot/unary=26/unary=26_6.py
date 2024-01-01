
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose1d(2, 3, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose1d(3, 4, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = self.conv_t2(t1)
        t3 = t2 > 0
        t4 = t2 * self.negative_slope
        t5 = torch.where(t3, t2, t4)
        return t5
negative_slope = -0.1
# Inputs to the model
x = torch.randn(2, 2, 1, 1)
