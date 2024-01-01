
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=1, padding=1)
    def forward(self, *input):
        t1 = self.conv(*input)
        t2 = t1 + 1
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        v6 = t4 / 6
        return v6
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=80, kernel_size=4, stride=1, padding=2)
    def forward(self, *input):
        t1 = self.conv(*input)
        t2 = t1 + 1
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        v6 = t4 / 6
        return v6
# Inputs to the model
x1 = torch.randn(3, 3, 32, 32)
