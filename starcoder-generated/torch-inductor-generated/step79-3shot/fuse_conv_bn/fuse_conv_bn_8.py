
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        for i in range(2):
            for j in range(3):
                setattr(self, "conv_{}".format(3*i + j), torch.nn.Conv2d(i * 5 + j, 3, 1))
                self.add_module("bn_{}".format(3*i + j), torch.nn.BatchNorm2d(3))
    def forward(self, x1):
        s1 = self.conv_0(x1)
        s1 = self.bn_0(s1)
        s1 = self.conv_1(s1)
        s1 = self.bn_1(s1)
        s1 = self.conv_2(s1)
        s1 = self.bn_2(s1)
        s4 = self.conv_5(x1)
        s4 = self.bn_5(s4)
        s4 = self.conv_6(s4)
        s4 = self.bn_6(s4)
        s4 = self.conv_7(s4)
        s4 = self.bn_7(s4)
        y4 = s1 + s4
        return y4
# Inputs to the model
x1 = torch.randn(1, 14, 4, 4)
