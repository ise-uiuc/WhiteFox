
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)
        self.conv_a = torch.nn.Conv2d(6, 48, 1, stride=1, padding=1)
        self.conv_b = torch.nn.Conv2d(48, 48, 1, stride=1, padding=0)
        self.conv_c = torch.nn.Conv2d(48, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.bn(x1)
        t2 = self.conv_a(t1)
        t3 = self.conv_b(t2)
        t4 = self.conv_c(t3)
        return t4
# Inputs to the model
x1 = torch.randn(1, 6, 128, 128)
