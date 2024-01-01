
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = 3 + self.conv(x1)
        t2 = self.relu6(t1)
        t3 = self.conv(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
