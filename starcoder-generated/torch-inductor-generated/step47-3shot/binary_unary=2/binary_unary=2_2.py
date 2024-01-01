
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_4_0 = torch.nn.Conv2d(1, 16, 4, stride=4, padding=0)
        self.conv_4_4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv_4_8 = torch.nn.Conv2d(16, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv_4_0(x1)
        t2 = self.conv_4_4(t1)
        t3 = self.conv_4_8(t2)
        t4 = t3
        return t4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
