
class m2(torch.nn.Module):
    def forward(self, x):
        tmp1 = x
        m1 = torch.nn.BatchNorm2d(4, affine=False)
        tmp2 = m1(tmp1)
        return tmp2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m2 = m2()
    def forward(self, x):
        tmp1 = x
        m1 = torch.nn.BatchNorm2d(4, affine=False)
        tmp2 = m1(tmp1)
        tmp3 = self.m2(tmp2)
        return tmp3
# Inputs to the model
x1 = torch.randn((2,2,2,2))
