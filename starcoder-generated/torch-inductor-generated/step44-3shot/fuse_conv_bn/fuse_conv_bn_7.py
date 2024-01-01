
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(8)
        self.m1 = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 2, bias=False), torch.nn.BatchNorm2d(2))
        self.m2 = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 2, bias=False), torch.nn.Sigmoid())
    def forward(self, x4):
        t = self.m1(x4)
        t2 = self.m2(t)
        return t2
# Inputs to the model
x4 = torch.randn(1, 2, 4, 4)
