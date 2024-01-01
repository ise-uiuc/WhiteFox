
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 32, 1, stride=1, padding=3)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        t3 = t2 + 3
        t4 = torch.clamp(t3, 0, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
