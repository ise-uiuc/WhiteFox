
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(11, 3, 2, stride=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)
        self.negative_slope = negative_slope
    def forward(self, input):
        t1 = self.conv_t1(input)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        return t8
negative_slope = (-8.4088, 5.1927)
# Inputs to the model
x1 = torch.randn(1, 11, 33, 19)
