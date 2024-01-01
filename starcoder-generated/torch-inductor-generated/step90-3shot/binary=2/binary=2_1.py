
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_2_0 = torch.nn.Conv2d(96, 96, (16, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=1, bias=False)
        self.conv_2_1 = torch.nn.Conv2d(96, 96, (16, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=1, bias=False)
        self.conv_2_2 = torch.nn.Conv2d(96, 96, (16, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=1, bias=False)
        self.conv_2_3 = torch.nn.Conv2d(96, 96, (16, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=1, bias=False)
    def forward(self, x):
        t0 = self.conv_2_0(F.pad(x, (10, 10, 5, 5), value=0.34790363216400146))
        t1 = self.conv_2_1(F.pad(x, (10, 10, 5, 5), value=-3.211524772644043))
        t2 = self.conv_2_2(F.pad(x, (10, 10, 5, 5), value=0.7422884187698364))
        t3 = self.conv_2_3(F.pad(x, (10, 10, 5, 5), value=-3.4418041229248047))
        t4 = t0 + -3
        t5 = t1 - 4.2186737060546875e-06
        t6 = t2 + -3.2397473335266113e-06
        t7 = t3 + 2.4659557342529297e-06
        return (t4, t5, t6, t7)
# Inputs to the model
x = torch.randn(1, 96, 112, 112)
