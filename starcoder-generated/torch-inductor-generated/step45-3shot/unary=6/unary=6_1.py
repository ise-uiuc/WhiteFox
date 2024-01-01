
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.maxpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv2d(x1)
        t2 = self.maxpool(t1)
        t3 = t2 / 11
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t1 * t5
        t7 = t6 /  6.0
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
