
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_d = 256
        self.layer = torch.nn.Conv2d(3, self.in_d - 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, momentum=0.)
    def forward(self, x):
        v1 = self.layer(x)
        v1 = self.bn(v1)
        v1, _ = torch.split(v1, (self.in_d - 3, 3), 1)
        return v1 + v1
# Inputs to the model
x = torch.randn(3, 3, 128, 128)
