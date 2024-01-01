
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(x1)
        t3 = 3 + t1
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t1 * t5
        t7 = t2 * t5
        t8 = (t6 + t7) / 2
        t9 = self.sigmoid(t8)
        return t9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
