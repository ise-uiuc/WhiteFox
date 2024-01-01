
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(101, 62, 2, stride=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(62, 62, 2, stride=1)
        self.relu_2 = torch.nn.ReLU6(True)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(relu_2(t6), t5, t7)
        return t8
        pass

negative_slope = 0.82
