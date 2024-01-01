
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(39, 256, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x):
        y = self.conv(x)
        z = self.conv2(y)
        p = torch.tanh(self.conv3(z))
        q = torch.tanh(self.conv4(p))
        r = torch.tanh(q)
        return r
# Inputs to the model
x = torch.randn(1, 39, 8, 4)
