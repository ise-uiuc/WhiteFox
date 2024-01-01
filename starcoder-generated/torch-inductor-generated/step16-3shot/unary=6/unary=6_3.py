
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 64, (1, 7), stride=1, padding=(0, 5), bias=False)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(x1)
        t3 = t1 + t2
        t4 = t3.clamp(0, 6)
        t5 = t1 - t2
        t6 = t5.clamp(0, 6)
        t7 = t4 * t6
        t8 = t7 / 6
        return t8
# Inputs to the model
x1 = torch.randn(1, 128, 64, 192)
