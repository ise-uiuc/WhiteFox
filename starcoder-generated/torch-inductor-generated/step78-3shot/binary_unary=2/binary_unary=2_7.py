
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 2, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 3, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 14, 3, stride=2, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t5 = torch.add(t2, 1.0)
        t3 = torch.mul(t5, 0.3)
        t4 = torch.div(t3, 1.2)
        t6 = torch.sub(t4, 1.5)
        y = self.conv3(t6)
        return y
# Inputs to the model
x1 = torch.randn(1, 14, 56, 56)
