
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module0 = Module0()
    def forward(self, x):
        v1 = self.module0(x)
        v2 = v1 - torch.tensor(32768)
        return v2
class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - -0.3086
        return v2
# Inputs to the model
x = torch.randn(1, 1, 27, 9)
