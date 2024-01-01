
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 4)
        self.batchnorm2d = torch.nn.BatchNorm2d(3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x0, x1):
        s = self.conv1(x0)
        t = self.batchnorm2d(s)
        u = self.tanh(t) + x1
        return t
# Inputs to the model
x0 = torch.rand((1, 3, 6, 6))
x1 = torch.rand((1, 3, 6, 6))
