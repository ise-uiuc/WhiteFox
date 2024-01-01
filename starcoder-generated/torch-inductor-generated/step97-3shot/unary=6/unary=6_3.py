
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(kernel_size=3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.pool(t1)
        t3 = self.relu(t2)
        t4 = self.conv2(t3)
        t5 = t4.clamp(min=0, max=6)
        t6 = t4 + 3
        t7 = t5 * t6
        t8 = t7 / 6
        return t8
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
