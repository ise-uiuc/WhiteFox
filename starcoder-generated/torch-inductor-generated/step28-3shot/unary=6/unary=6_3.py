
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.activ1 = torch.nn.ReLU()
        self.activ2 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x):
        v = self.conv(x)
        w = self.activ1(v)
        x = self.activ2(v)
        y = self.conv1(v)
        z = self.conv1(v)
        return (w, x, y, z)
# Inputs to the model
x = torch.randn(1, 3, 384, 384)
