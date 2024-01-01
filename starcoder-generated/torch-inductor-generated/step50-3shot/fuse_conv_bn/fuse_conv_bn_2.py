
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        w = self.conv3(t)
        return w
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
