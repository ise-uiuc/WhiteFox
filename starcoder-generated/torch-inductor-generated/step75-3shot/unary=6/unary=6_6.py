
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, dilation=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, dilation=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 1, 3, stride=1, dilation=1, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        return t3
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
